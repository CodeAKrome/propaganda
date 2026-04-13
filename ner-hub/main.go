// ner-processor.go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/schollz/progressbar/v3"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	dbName             = "rssnews"
	collName           = "articles"
	timeout            = 360 * time.Second
	workersPerEndpoint = 2 // Workers per endpoint for parallel processing
	maxTextLength      = 60000
	maxRetries         = 3
	retryDelay         = 2 * time.Second
	sharedQueueSize    = 100 // Shared queue buffer size
)

var errorLog *os.File

func init() {
	var err error
	errorLog, err = os.OpenFile("ner.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		log.Fatalf("cannot open ner.log: %v", err)
	}
	log.SetOutput(io.MultiWriter(os.Stderr, errorLog))
}

type Config struct {
	Endpoints []Endpoint
}

type Endpoint struct {
	Name string
	URL  string
}

type NERRequest struct {
	Text string `json:"text"`
}

type EntityInfo struct {
	Text  string `json:"text"`
	Label string `json:"label"`
}

type NERResponse struct {
	Entities      []EntityInfo   `json:"entities"`
	EntityCounts  map[string]int `json:"entity_counts"`
	TotalEntities int            `json:"total_entities"`
}

type ProcessJob struct {
	ID      interface{}
	Article string
}

type Stats struct {
	mu               sync.Mutex
	EndpointName     string
	Processed        int
	Errors           int
	Skipped          int
	AlreadyProcessed int
	TotalRequests    int64
	TotalLatency     time.Duration
	MinLatency       time.Duration
	MaxLatency       time.Duration
	EntityCounts     map[string]int64 // Track entity types
}

func (s *Stats) IncrProcessed() {
	s.mu.Lock()
	s.Processed++
	s.mu.Unlock()
}

func (s *Stats) IncrErrors() {
	s.mu.Lock()
	s.Errors++
	s.mu.Unlock()
}

func (s *Stats) IncrSkipped() {
	s.mu.Lock()
	s.Skipped++
	s.mu.Unlock()
}

func (s *Stats) IncrAlreadyProcessed() {
	s.mu.Lock()
	s.AlreadyProcessed++
	s.mu.Unlock()
}

func (s *Stats) RecordLatency(d time.Duration) {
	s.mu.Lock()
	s.TotalRequests++
	s.TotalLatency += d
	if s.MinLatency == 0 || d < s.MinLatency {
		s.MinLatency = d
	}
	if d > s.MaxLatency {
		s.MaxLatency = d
	}
	s.mu.Unlock()
}

func (s *Stats) RecordEntities(entities []EntityInfo) {
	s.mu.Lock()
	if s.EntityCounts == nil {
		s.EntityCounts = make(map[string]int64)
	}
	for _, entity := range entities {
		s.EntityCounts[entity.Label]++
	}
	s.mu.Unlock()
}

func (s *Stats) GetCounts() (int, int, int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.Processed, s.Errors, s.Skipped, s.AlreadyProcessed
}

func (s *Stats) GetLatencyStats() (time.Duration, time.Duration, time.Duration, int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var avg time.Duration
	if s.TotalRequests > 0 {
		avg = s.TotalLatency / time.Duration(s.TotalRequests)
	}
	return s.MinLatency, avg, s.MaxLatency, s.TotalRequests
}

func (s *Stats) GetEntityCounts() map[string]int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	result := make(map[string]int64)
	for k, v := range s.EntityCounts {
		result[k] = v
	}
	return result
}

type WorkerPool struct {
	endpoint Endpoint
	stats    *Stats
	coll     *mongo.Collection
}

type GlobalStats struct {
	StartTime time.Time
	EndTime   time.Time
}

var (
	globalStats  GlobalStats
	shutdownFlag int32
)

func main() {
	defer errorLog.Close()

	var startDate, endDate string
	var configPath string

	flag.StringVar(&configPath, "config", "", "Config file path (required)")
	flag.StringVar(&startDate, "start-date", "", "Start date (YYYY-MM-DD or relative like -7 for 7 days ago)")
	flag.StringVar(&endDate, "end-date", "", "End date (YYYY-MM-DD or relative like -1 for yesterday)")
	flag.Parse()

	if configPath == "" {
		args := flag.Args()
		if len(args) == 1 {
			configPath = args[0]
		} else {
			log.Fatalf("usage: %s [--start-date DATE] [--end-date DATE] <endpoints.tsv>", os.Args[0])
		}
	}

	var startTime, endTime *time.Time
	var err error

	if startDate != "" {
		startTime, err = parseDate(startDate)
		if err != nil {
			log.Fatalf("invalid start-date: %v", err)
		}
	}

	if endDate != "" {
		endTime, err = parseDate(endDate)
		if err != nil {
			log.Fatalf("invalid end-date: %v", err)
		}
		*endTime = endTime.Add(24*time.Hour - time.Second)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	uri := "mongodb://" + os.Getenv("MONGO_USER") + ":" + os.Getenv("MONGO_PASS") + "@localhost:27017"
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		log.Fatalf("mongo connect: %v", err)
	}
	defer client.Disconnect(ctx)

	coll := client.Database(dbName).Collection(collName)

	config, err := loadConfig(configPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	if len(config.Endpoints) == 0 {
		log.Fatalf("no endpoints defined in config")
	}

	log.Printf("Loaded %d endpoint(s)", len(config.Endpoints))
	for _, ep := range config.Endpoints {
		log.Printf("  - %s: %s", ep.Name, ep.URL)
	}

	filter := bson.M{
		"article":     bson.M{"$exists": true, "$ne": nil},
		"fetch_error": bson.M{"$exists": false},
		"ner":         bson.M{"$exists": false},
	}

	if startTime != nil || endTime != nil {
		dateFilter := bson.M{}
		if startTime != nil {
			dateFilter["$gte"] = *startTime
			log.Printf("Filtering from: %s", startTime.Format("2006-01-02"))
		}
		if endTime != nil {
			dateFilter["$lte"] = *endTime
			log.Printf("Filtering to: %s", endTime.Format("2006-01-02"))
		}
		filter["published"] = dateFilter
	}

	total, err := coll.CountDocuments(ctx, filter)
	if err != nil {
		log.Fatalf("count documents: %v", err)
	}
	log.Printf("Found %d articles to process", total)

	if total == 0 {
		log.Println("No articles to process")
		return
	}

	// Initialize global stats
	globalStats.StartTime = time.Now()

	// Create progress bar
	bar := progressbar.NewOptions(int(total),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowCount(),
		progressbar.OptionSetWidth(40),
		progressbar.OptionSetDescription("[cyan]Processing articles[reset]"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[green]=[reset]",
			SaucerHead:    "[green]>[reset]",
			SaucerPadding: " ",
			BarStart:      "[",
			BarEnd:        "]",
		}),
		progressbar.OptionOnCompletion(func() {
			fmt.Fprint(os.Stderr, "\n")
		}),
	)

	// Create a SINGLE shared job queue
	sharedJobs := make(chan ProcessJob, sharedQueueSize)

	// Create worker pools for each endpoint
	pools := make([]*WorkerPool, len(config.Endpoints))
	var wg sync.WaitGroup

	for i, ep := range config.Endpoints {
		pools[i] = &WorkerPool{
			endpoint: ep,
			stats:    &Stats{EndpointName: ep.Name, EntityCounts: make(map[string]int64)},
			coll:     coll,
		}

		// Start workers for this endpoint - all reading from the SAME shared queue
		for w := 0; w < workersPerEndpoint; w++ {
			wg.Add(1)
			go worker(ctx, pools[i], sharedJobs, bar, &wg)
		}
	}

	// Fetch and send articles to the shared queue (only once per article)
	fetchDone := make(chan struct{})
	go func() {
		defer close(fetchDone)
		cur, err := coll.Find(ctx, filter, options.Find().SetProjection(bson.M{"_id": 1, "article": 1}))
		if err != nil {
			log.Fatalf("find articles: %v", err)
		}
		defer cur.Close(ctx)

		jobCount := 0
		for cur.Next(ctx) {
			// Check if shutdown was requested
			if atomic.LoadInt32(&shutdownFlag) == 1 {
				log.Println("\nShutdown requested, stopping article fetching...")
				break
			}

			var doc struct {
				ID      interface{} `bson:"_id"`
				Article *string     `bson:"article"`
			}
			if err := cur.Decode(&doc); err != nil {
				log.Printf("decode error: %v", err)
				continue
			}
			if doc.Article == nil || *doc.Article == "" {
				continue
			}

			job := ProcessJob{ID: doc.ID, Article: *doc.Article}
			// Send to shared queue - only ONE worker will pick it up
			select {
			case sharedJobs <- job:
				jobCount++
			case <-ctx.Done():
				log.Println("Context cancelled, stopping article fetching...")
				return
			}
		}

		// Close the shared job channel after all jobs are sent
		close(sharedJobs)

		if err := cur.Err(); err != nil {
			log.Printf("cursor error: %v", err)
		}
	}()

	// Handle shutdown signal
	go func() {
		<-sigChan
		log.Println("\n\nReceived interrupt signal (Ctrl-C). Shutting down gracefully...")
		atomic.StoreInt32(&shutdownFlag, 1)
		cancel() // Cancel context to stop workers gracefully
	}()

	// Wait for either completion or shutdown
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Normal completion
		<-fetchDone // Ensure fetching is done
	case <-ctx.Done():
		// Shutdown requested
		<-fetchDone // Wait for fetch goroutine to stop
		// Wait for workers with timeout
		waitChan := make(chan struct{})
		go func() {
			wg.Wait()
			close(waitChan)
		}()

		select {
		case <-waitChan:
			log.Println("All workers stopped gracefully")
		case <-time.After(5 * time.Second):
			log.Println("Timeout waiting for workers, forcing shutdown")
		}
	}

	globalStats.EndTime = time.Now()

	// Ensure progress bar is complete
	bar.Finish()

	// Print detailed statistics report
	printStatisticsReport(pools)
}

func printStatisticsReport(pools []*WorkerPool) {
	uptime := globalStats.EndTime.Sub(globalStats.StartTime)
	uptimeStr := formatDuration(uptime)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("NER PROCESSOR - SHUTDOWN STATISTICS REPORT")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("Shutdown Time:        %s\n", globalStats.EndTime.Format("2006-01-02 15:04:05"))
	fmt.Printf("Total Runtime:        %s\n", uptimeStr)
	fmt.Println(strings.Repeat("-", 70))

	totalProcessed := 0
	totalErrors := 0
	totalSkipped := 0
	totalAlreadyProcessed := 0
	totalRequests := int64(0)
	totalLatency := time.Duration(0)
	globalMinLatency := time.Duration(0)
	globalMaxLatency := time.Duration(0)
	globalEntityCounts := make(map[string]int64)

	for _, pool := range pools {
		processed, errors, skipped, alreadyProcessed := pool.stats.GetCounts()
		minLat, avgLat, maxLat, reqCount := pool.stats.GetLatencyStats()
		entityCounts := pool.stats.GetEntityCounts()

		fmt.Printf("\n%-20s\n", pool.stats.EndpointName+":")
		fmt.Printf("  Processed:          %d\n", processed)
		fmt.Printf("  Already processed:  %d\n", alreadyProcessed)
		fmt.Printf("  Errors:             %d\n", errors)
		fmt.Printf("  Skipped (too long): %d\n", skipped)

		if reqCount > 0 {
			fmt.Printf("  Requests sent:      %d\n", reqCount)
			fmt.Printf("  Latency (min/avg/max): %v / %v / %v\n",
				minLat.Round(time.Millisecond),
				avgLat.Round(time.Millisecond),
				maxLat.Round(time.Millisecond))
		}

		totalProcessed += processed
		totalErrors += errors
		totalSkipped += skipped
		totalAlreadyProcessed += alreadyProcessed
		totalRequests += reqCount
		totalLatency += pool.stats.TotalLatency

		if globalMinLatency == 0 || (minLat > 0 && minLat < globalMinLatency) {
			globalMinLatency = minLat
		}
		if maxLat > globalMaxLatency {
			globalMaxLatency = maxLat
		}

		// Aggregate entity counts
		for label, count := range entityCounts {
			globalEntityCounts[label] += count
		}
	}

	fmt.Printf("\n%s\n", strings.Repeat("-", 70))
	fmt.Println("OVERALL TOTALS:")
	fmt.Printf("  Newly processed:    %d\n", totalProcessed)
	fmt.Printf("  Already processed:  %d\n", totalAlreadyProcessed)
	fmt.Printf("  Errors:             %d\n", totalErrors)
	fmt.Printf("  Skipped (too long): %d\n", totalSkipped)
	fmt.Printf("  Total articles:     %d\n", totalProcessed+totalAlreadyProcessed+totalErrors+totalSkipped)

	if totalRequests > 0 {
		avgLatency := totalLatency / time.Duration(totalRequests)
		fmt.Printf("\n  Total requests:     %d\n", totalRequests)
		fmt.Printf("  Global latency (min/avg/max): %v / %v / %v\n",
			globalMinLatency.Round(time.Millisecond),
			avgLatency.Round(time.Millisecond),
			globalMaxLatency.Round(time.Millisecond))

		if uptime > 0 {
			articlesPerSec := float64(totalProcessed) / uptime.Seconds()
			fmt.Printf("  Processing rate:    %.2f articles/second\n", articlesPerSec)
		}
	}

	if len(globalEntityCounts) > 0 {
		fmt.Printf("\n%s\n", strings.Repeat("-", 70))
		fmt.Println("ENTITY TYPE DISTRIBUTION:")

		// Sort entity types by count
		type entityCount struct {
			Label string
			Count int64
		}
		var sorted []entityCount
		totalEntities := int64(0)
		for label, count := range globalEntityCounts {
			sorted = append(sorted, entityCount{Label: label, Count: count})
			totalEntities += count
		}

		// Simple bubble sort
		for i := 0; i < len(sorted)-1; i++ {
			for j := 0; j < len(sorted)-i-1; j++ {
				if sorted[j].Count < sorted[j+1].Count {
					sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
				}
			}
		}

		for _, ec := range sorted {
			percentage := float64(ec.Count) / float64(totalEntities) * 100
			fmt.Printf("  %-20s %8d entities (%5.2f%%)\n", ec.Label, ec.Count, percentage)
		}
		fmt.Printf("\n  Total entities:     %d\n", totalEntities)
	}

	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()
}

func formatDuration(d time.Duration) string {
	hours := int(d.Hours())
	minutes := int(d.Minutes()) % 60
	seconds := int(d.Seconds()) % 60
	return fmt.Sprintf("%02d:%02d:%02d", hours, minutes, seconds)
}

func worker(ctx context.Context, pool *WorkerPool, jobs <-chan ProcessJob, bar *progressbar.ProgressBar, wg *sync.WaitGroup) {
	defer wg.Done()
	client := &http.Client{Timeout: timeout}

	for {
		select {
		case <-ctx.Done():
			// Context cancelled, stop processing
			return
		case job, ok := <-jobs:
			if !ok {
				// Channel closed, no more jobs
				return
			}

			// Check if already processed
			var exists struct {
				ID  interface{}  `bson:"_id"`
				NER *NERResponse `bson:"ner"`
			}
			err := pool.coll.FindOne(ctx,
				bson.M{"_id": job.ID},
				options.FindOne().SetProjection(bson.M{"_id": 1, "ner": 1}),
			).Decode(&exists)

			if err == nil && exists.NER != nil {
				// Already processed
				pool.stats.IncrAlreadyProcessed()
				bar.Add(1)
				continue
			}

			// Process the article
			start := time.Now()
			nerResult, skipped, err := callNERServiceLog(client, pool.endpoint, job.Article)
			latency := time.Since(start)

			if err != nil {
				if skipped {
					update := bson.M{
						"$set": bson.M{
							"ner_skipped": "text_too_long",
						},
					}
					updateOneLog(ctx, pool.coll, pool.endpoint, job.ID, update)
					pool.stats.IncrSkipped()
				} else {
					pool.stats.IncrErrors()
				}
				bar.Add(1)
				continue
			}

			// Record successful request latency
			pool.stats.RecordLatency(latency)

			// Record entity statistics
			if nerResult != nil {
				pool.stats.RecordEntities(nerResult.Entities)
			}

			// Update the document
			update := bson.M{
				"$set": bson.M{
					"ner": nerResult,
				},
			}

			_, err = pool.coll.UpdateOne(ctx, bson.M{"_id": job.ID}, update)

			if err != nil {
				pool.stats.IncrErrors()
				reportError(pool.endpoint.Name, "mongoUpdate", err, "")
			} else {
				pool.stats.IncrProcessed()
			}

			bar.Add(1)
		}
	}
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	lines := strings.Split(string(data), "\n")
	config := &Config{}

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid line: %q", line)
		}
		name := strings.TrimSpace(parts[0])
		url := strings.TrimSpace(parts[1])
		config.Endpoints = append(config.Endpoints, Endpoint{Name: name, URL: url})
	}

	return config, nil
}

func parseDate(dateStr string) (*time.Time, error) {
	if strings.HasPrefix(dateStr, "-") {
		days, err := strconv.Atoi(dateStr)
		if err != nil {
			return nil, fmt.Errorf("invalid relative date: %s", dateStr)
		}
		t := time.Now().AddDate(0, 0, days)
		t = time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
		return &t, nil
	}

	t, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		return nil, fmt.Errorf("invalid date format (use YYYY-MM-DD or -N): %s", dateStr)
	}
	return &t, nil
}

func reportError(server, op string, err error, extra string) {
	buf, _ := json.Marshal(struct {
		Server    string `json:"server"`
		Operation string `json:"operation"`
		Error     string `json:"error"`
	}{
		Server:    server,
		Operation: op,
		Error:     err.Error(),
	})
	log.Printf("ner_error %s  %s", string(buf), extra)
}

func callNERServiceLog(client *http.Client, ep Endpoint, text string) (*NERResponse, bool, error) {
	res, skipped, err := callNERService(client, ep.URL, text)
	if err != nil && !skipped {
		reportError(ep.Name, "callNERService", err, "")
	}
	return res, skipped, err
}

func updateOneLog(ctx context.Context, coll *mongo.Collection, ep Endpoint, id interface{}, update bson.M) error {
	_, err := coll.UpdateOne(ctx, bson.M{"_id": id}, update)
	if err != nil {
		reportError(ep.Name, "mongoUpdate", err, "")
	}
	return err
}

func callNERService(client *http.Client, baseURL, text string) (*NERResponse, bool, error) {
	skipped := false
	if len(text) > maxTextLength {
		log.Printf("Article too long (%d chars), skipping NER processing", len(text))
		return nil, true, fmt.Errorf("text too long: %d characters", len(text))
	}

	reqBody := NERRequest{Text: text}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, skipped, fmt.Errorf("marshal request: %w", err)
	}

	url := baseURL + "/extract"

	var lastErr error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, skipped, fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			if attempt < maxRetries {
				if strings.Contains(err.Error(), "connection reset") ||
					strings.Contains(err.Error(), "EOF") ||
					strings.Contains(err.Error(), "broken pipe") {
					log.Printf("Attempt %d/%d failed for %s: %v (retrying after %v)",
						attempt, maxRetries, baseURL, err, retryDelay)
					time.Sleep(retryDelay)
					continue
				}
			}
			return nil, skipped, fmt.Errorf("http request: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			bodyStr := string(body)

			if resp.StatusCode >= 500 && attempt < maxRetries {
				log.Printf("Attempt %d/%d failed with HTTP %d for %s (retrying after %v)",
					attempt, maxRetries, resp.StatusCode, baseURL, retryDelay)
				time.Sleep(retryDelay)
				continue
			}

			reportError(baseURL, "httpNon200",
				fmt.Errorf("HTTP %d", resp.StatusCode),
				fmt.Sprintf("body=%q", bodyStr))
			return nil, skipped, fmt.Errorf("HTTP %d: %s", resp.StatusCode, bodyStr)
		}

		var nerResp NERResponse
		if err := json.NewDecoder(resp.Body).Decode(&nerResp); err != nil {
			return nil, skipped, fmt.Errorf("decode response: %w", err)
		}

		return &nerResp, skipped, nil
	}

	return nil, skipped, fmt.Errorf("all %d retry attempts failed, last error: %w", maxRetries, lastErr)
}
