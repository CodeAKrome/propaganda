// main.go - RSS feed aggregator with user agent rotation and curl fallback
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/mmcdole/gofeed"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"golang.org/x/net/html"
)

const (
	dbName         = "rssnews"
	collName       = "articles"
	statsCollName  = "stats"
	statsDocID     = "stats"
	workerCount    = 8
	requestTimeout = 15 * time.Second
	maxRetries     = 3
	initialBackoff = 2 * time.Second
	MINLINE        = 128
	uaStatsFile    = "user_agent_stats.txt"
)

var userAgents = []string{
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.7843.90 Safari/537.36",
	"Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.7843.90 Mobile Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
	"Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/605.1.15",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 14.6; rv:132.0) Gecko/20100101 Firefox/132.0",
}

type Article struct {
	Source      string    `bson:"source"`
	Title       string    `bson:"title"`
	Description string    `bson:"description"`
	Link        string    `bson:"link"`
	Published   time.Time `bson:"published"`
	Raw         *string   `bson:"raw,omitempty"`
	Article     *string   `bson:"article,omitempty"`
	FetchError  *string   `bson:"fetch_error,omitempty"`
	Tags        []string  `bson:"tags"`
}

type Stats struct {
	ID           string         `bson:"_id"`
	SourceCounts map[string]int `bson:"source_counts"`
	Updated      time.Time      `bson:"updated"`
}

type UAStats struct {
	mu      sync.RWMutex
	success map[string]int // user agent -> success count
	failure map[string]int // user agent -> failure count
}

var (
	sourceRegex = make(map[string]*regexp.Regexp)
	uaStats     = &UAStats{
		success: make(map[string]int),
		failure: make(map[string]int),
	}
)

func (u *UAStats) load() error {
	u.mu.Lock()
	defer u.mu.Unlock()

	f, err := os.Open(uaStatsFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // file doesn't exist yet, that's ok
		}
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "\t", 3)
		if len(parts) != 3 {
			continue
		}
		ua := parts[0]
		var succ, fail int
		fmt.Sscanf(parts[1], "%d", &succ)
		fmt.Sscanf(parts[2], "%d", &fail)
		u.success[ua] = succ
		u.failure[ua] = fail
	}
	return sc.Err()
}

func (u *UAStats) save() error {
	u.mu.RLock()
	defer u.mu.RUnlock()

	f, err := os.Create(uaStatsFile)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	w.WriteString("# User Agent\tSuccess\tFailure\n")
	for _, ua := range userAgents {
		succ := u.success[ua]
		fail := u.failure[ua]
		fmt.Fprintf(w, "%s\t%d\t%d\n", ua, succ, fail)
	}
	return w.Flush()
}

func (u *UAStats) recordSuccess(ua string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.success[ua]++
}

func (u *UAStats) recordFailure(ua string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.failure[ua]++
}

// getSortedUserAgents returns user agents sorted by success rate (best first)
func (u *UAStats) getSortedUserAgents() []string {
	u.mu.RLock()
	defer u.mu.RUnlock()

	type uaScore struct {
		ua    string
		score float64
	}

	scores := make([]uaScore, len(userAgents))
	for i, ua := range userAgents {
		succ := float64(u.success[ua])
		fail := float64(u.failure[ua])
		total := succ + fail

		// Score: success rate, but add small bonus for being tried
		// Default to original order if never tried
		score := float64(i) / 100.0 // small tiebreaker for original order
		if total > 0 {
			score = succ / total
		}
		scores[i] = uaScore{ua: ua, score: score}
	}

	// Sort by score descending
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	result := make([]string, len(scores))
	for i, s := range scores {
		result[i] = s.ua
	}
	return result
}

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) != 2 {
		log.Fatalf("usage: %s <feeds.tsv> <clean-rules.tsv>", os.Args[0])
	}
	cfgPath := args[0]
	rulesPath := args[1]

	// Load user agent stats
	if err := uaStats.load(); err != nil {
		log.Printf("warning: could not load UA stats: %v", err)
	}

	ctx := context.Background()
	uri := "mongodb://" + os.Getenv("MONGO_USER") + ":" + os.Getenv("MONGO_PASS") + "@localhost:27017"
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		log.Fatalf("mongo connect: %v", err)
	}
	defer client.Disconnect(ctx)

	articlesColl := client.Database(dbName).Collection(collName)
	statsColl := client.Database(dbName).Collection(statsCollName)

	// 1. read feeds config
	sources, err := readConfig(cfgPath)
	if err != nil {
		log.Fatalf("read config: %v", err)
	}

	// 2. load regex rules
	if err := loadRegexRules(rulesPath); err != nil {
		log.Fatalf("load rules: %v", err)
	}

	// 3. fetch RSS
	feeds := fetchAllFeeds(sources)

	// 4. store articles (raw + cleaned)  +  print per-source summary
	type sum struct{ added, errs, skipped int }
	perSource := make(map[string]sum)

	for _, a := range feeds {
		perSource[a.Source] = sum{} // ensure every source is listed
	}

	for src := range perSource {
		var arts []Article
		for _, a := range feeds {
			if a.Source == src {
				arts = append(arts, a)
			}
		}
		add, errn, skip, err := storeArticles(ctx, articlesColl, arts)
		if err != nil {
			log.Fatalf("store articles for %s: %v", src, err)
		}
		perSource[src] = sum{added: add, errs: errn, skipped: skip}
	}

	for src, s := range perSource {
		fmt.Printf("✅ %-20s  added=%-4d  fetch-errors=%-4d  skipped=%d\n", src, s.added, s.errs, s.skipped)
	}

	// 5. backfill raw & article if missing
	if err := backfillArticles(ctx, articlesColl); err != nil {
		log.Fatalf("backfill: %v", err)
	}

	// 6. update stats
	if err := updateStats(ctx, articlesColl, statsColl); err != nil {
		log.Fatalf("update stats: %v", err)
	}

	// 7. save user agent stats
	if err := uaStats.save(); err != nil {
		log.Printf("warning: could not save UA stats: %v", err)
	}

	log.Println("all done")
}

/* ---------- helpers ---------- */

func readConfig(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	out := make(map[string]string)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid line: %q", line)
		}
		name, url := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		out[name] = url
	}
	return out, sc.Err()
}

func loadRegexRules(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid rule line: %q", line)
		}
		src, pat := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		// (?s) = single-line mode → dot matches newline
		re, err := regexp.Compile("(?s)" + pat)
		if err != nil {
			return fmt.Errorf("bad regex for %s: %w", src, err)
		}
		sourceRegex[src] = re
	}
	return sc.Err()
}

func cleanText(src, raw string) string {
	re, ok := sourceRegex[src]
	if !ok {
		return raw
	}
	return re.ReplaceAllString(raw, "")
}

// fetchWithCurl uses curl as a fallback when all user agents fail
func fetchWithCurl(url string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "curl", "-s", "-L", url)
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("curl failed: %w", err)
	}
	return string(output), nil
}

// parseFeedWithRetry tries user agents in order, then falls back to curl
func parseFeedWithRetry(fp *gofeed.Parser, name, url string, maxRetries int) (*gofeed.Feed, error) {
	sortedUAs := uaStats.getSortedUserAgents()

	// Try each user agent in order
	for i, ua := range sortedUAs {
		client := &http.Client{
			Timeout: requestTimeout,
		}
		fp.Client = client
		fp.UserAgent = ua

		feed, err := fp.ParseURL(url)
		if err == nil {
			uaStats.recordSuccess(ua)
			if i > 0 {
				log.Printf("  ✅ %s succeeded with UA #%d", name, i+1)
			}
			return feed, nil
		}

		// Check if it's a retriable error
		if strings.Contains(err.Error(), "403") ||
			strings.Contains(err.Error(), "429") ||
			strings.Contains(err.Error(), "Forbidden") ||
			strings.Contains(err.Error(), "Too Many Requests") {
			uaStats.recordFailure(ua)
			log.Printf("  ⚠️  %s failed with UA #%d: %v", name, i+1, err)
			continue
		}

		// Non-retriable error, return it
		return nil, err
	}

	// All user agents failed, try curl
	log.Printf("  🔄 %s: all user agents failed, trying curl...", name)
	body, err := fetchWithCurl(url)
	if err != nil {
		return nil, fmt.Errorf("all methods failed, curl: %w", err)
	}

	// Parse the curl output as RSS/Atom
	feed, err := fp.ParseString(body)
	if err != nil {
		return nil, fmt.Errorf("curl succeeded but parse failed: %w", err)
	}

	log.Printf("  ✅ %s succeeded with curl", name)
	return feed, nil
}

func fetchAllFeeds(src map[string]string) []Article {
	var (
		mu  sync.Mutex
		out []Article
		wg  sync.WaitGroup
	)

	for name, url := range src {
		wg.Add(1)
		go func(n, u string) {
			defer wg.Done()
			fp := gofeed.NewParser()
			feed, err := parseFeedWithRetry(fp, n, u, maxRetries)
			if err != nil {
				log.Printf("⚠️  feed failed: %s → %v", n, err)
				return
			}
			mu.Lock()
			for _, item := range feed.Items {
				a := Article{
					Source:      n,
					Title:       item.Title,
					Description: item.Description,
					Link:        item.Link,
				}
				if item.PublishedParsed != nil {
					a.Published = *item.PublishedParsed
				} else {
					a.Published = time.Now()
				}
				out = append(out, a)
			}
			mu.Unlock()
			fmt.Printf("✅ %s\n", n)
		}(name, url)
	}
	wg.Wait()
	return out
}

// fetchArticleWithUserAgents tries user agents in order, then curl
func fetchArticleWithRetry(url string, maxRetries int) (string, error) {
	sortedUAs := uaStats.getSortedUserAgents()

	// Try each user agent in order
	for _, ua := range sortedUAs {
		body, err := fetchArticleWithUA(url, ua)
		if err == nil {
			uaStats.recordSuccess(ua)
			return body, nil
		}

		// Check if it's a retriable error
		if strings.Contains(err.Error(), "403") ||
			strings.Contains(err.Error(), "429") ||
			strings.Contains(err.Error(), "Forbidden") ||
			strings.Contains(err.Error(), "Too Many Requests") {
			uaStats.recordFailure(ua)
			continue
		}

		// Non-retriable error, return it
		return "", err
	}

	// All user agents failed, try curl
	body, err := fetchWithCurl(url)
	if err != nil {
		return "", fmt.Errorf("all methods failed: %w", err)
	}

	// Parse HTML from curl output
	return parseHTMLBody(body)
}

func fetchArticleWithUA(url, userAgent string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("http %d", resp.StatusCode)
	}

	root, err := html.Parse(resp.Body)
	if err != nil {
		return "", err
	}

	return extractParagraphs(root)
}

func parseHTMLBody(htmlContent string) (string, error) {
	root, err := html.Parse(strings.NewReader(htmlContent))
	if err != nil {
		return "", err
	}
	return extractParagraphs(root)
}

func extractParagraphs(root *html.Node) (string, error) {
	var paras []string
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "p" {
			var sb strings.Builder
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				renderText(c, &sb)
			}
			if txt := strings.TrimSpace(sb.String()); txt != "" {
				paras = append(paras, txt)
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(root)

	if len(paras) == 0 {
		return "", errors.New("no <p> content found")
	}
	return strings.Join(paras, "\n\n"), nil
}

func isDuplicateKeyError(err error) bool {
	if err == nil {
		return false
	}
	if mongo.IsDuplicateKeyError(err) {
		return true
	}
	return strings.Contains(err.Error(), "E11000") || strings.Contains(err.Error(), "duplicate key")
}

func storeArticles(ctx context.Context, coll *mongo.Collection, arts []Article) (int, int, int, error) {
	links := make([]string, len(arts))
	for i, a := range arts {
		links[i] = a.Link
	}

	cur, err := coll.Find(ctx, bson.M{"link": bson.M{"$in": links}}, options.Find().SetProjection(bson.M{"link": 1}))
	if err != nil {
		return 0, 0, 0, err
	}

	existing := make(map[string]bool)
	for cur.Next(ctx) {
		var doc struct {
			Link string `bson:"link"`
		}
		if err := cur.Decode(&doc); err == nil {
			existing[doc.Link] = true
		}
	}
	cur.Close(ctx)

	var added, errs, skipped int
	for _, a := range arts {
		if existing[a.Link] {
			skipped++
			continue
		}

		body, err := fetchArticleWithRetry(a.Link, maxRetries)
		if err != nil {
			msg := err.Error()
			a.FetchError = &msg
			errs++
		} else {
			raw := body
			article := cleanText(a.Source, body)

			if len(article) < MINLINE {
				skipped++
				continue
			}

			a.Raw = &raw
			a.Article = &article
		}

		a.Tags = []string{}

		_, err = coll.InsertOne(ctx, a)
		if err != nil {
			if isDuplicateKeyError(err) {
				skipped++
				continue
			}
			return added, errs, skipped, err
		}
		added++
	}

	return added, errs, skipped, nil
}

func backfillArticles(ctx context.Context, coll *mongo.Collection) error {
	cur, err := coll.Find(ctx, bson.M{"raw": bson.M{"$exists": false}})
	if err != nil {
		return err
	}
	defer cur.Close(ctx)

	type job struct {
		id     interface{}
		url    string
		source string
	}
	jobs := make(chan job, 100)
	var wg sync.WaitGroup

	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				body, err := fetchArticleWithRetry(j.url, maxRetries)
				var update bson.M
				if err != nil {
					msg := err.Error()
					update = bson.M{"$set": bson.M{"fetch_error": msg}}
				} else {
					raw := body
					article := cleanText(j.source, body)

					if len(article) < MINLINE {
						msg := fmt.Sprintf("article too short: %d chars (minimum %d)", len(article), MINLINE)
						update = bson.M{"$set": bson.M{"fetch_error": msg}}
					} else {
						update = bson.M{"$set": bson.M{
							"raw":         raw,
							"article":     article,
							"fetch_error": nil,
						}}
					}
				}
				_, _ = coll.UpdateOne(ctx, bson.M{"_id": j.id}, update)
			}
		}()
	}

	for cur.Next(ctx) {
		var doc struct {
			ID     interface{} `bson:"_id"`
			Link   string      `bson:"link"`
			Source string      `bson:"source"`
		}
		if err := cur.Decode(&doc); err != nil {
			continue
		}
		jobs <- job{id: doc.ID, url: doc.Link, source: doc.Source}
	}
	close(jobs)
	wg.Wait()
	return cur.Err()
}

func fetchArticle(url string) (string, error) {
	sortedUAs := uaStats.getSortedUserAgents()
	if len(sortedUAs) > 0 {
		return fetchArticleWithUA(url, sortedUAs[0])
	}
	return fetchArticleWithUA(url, userAgents[0])
}

func renderText(n *html.Node, sb *strings.Builder) {
	if n.Type == html.TextNode {
		sb.WriteString(n.Data)
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		renderText(c, sb)
	}
}

func updateStats(ctx context.Context, artColl, statsColl *mongo.Collection) error {
	match := bson.M{"$group": bson.M{"_id": "$source", "count": bson.M{"$sum": 1}}}
	cur, err := artColl.Aggregate(ctx, []bson.M{{"$match": bson.M{}}, match})
	if err != nil {
		return err
	}
	defer cur.Close(ctx)

	counts := make(map[string]int)
	for cur.Next(ctx) {
		var row struct {
			ID    string `bson:"_id"`
			Count int    `bson:"count"`
		}
		if err := cur.Decode(&row); err != nil {
			continue
		}
		counts[row.ID] = row.Count
	}

	_, err = statsColl.ReplaceOne(ctx,
		bson.M{"_id": statsDocID},
		Stats{ID: statsDocID, SourceCounts: counts, Updated: time.Now()},
		options.Replace().SetUpsert(true))
	return err
}
