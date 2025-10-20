// main.go  (dot-all flag added → deletions cross new-lines)
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
)

type Article struct {
	Source      string    `bson:"source"`
	Title       string    `bson:"title"`
	Description string    `bson:"description"`
	Link        string    `bson:"link"`
	Published   time.Time `bson:"published"`
	Raw         *string   `bson:"raw,omitempty"`     // original scraped text
	Article     *string   `bson:"article,omitempty"` // cleaned text
	FetchError  *string   `bson:"fetch_error,omitempty"`
	Tags        []string  `bson:"tags"` // <-- NEW: always init to []
}

type Stats struct {
	ID           string         `bson:"_id"`
	SourceCounts map[string]int `bson:"source_counts"`
	Updated      time.Time      `bson:"updated"`
}

// source-specific regex cleaners
var sourceRegex = make(map[string]*regexp.Regexp)

// --- 2.  main collects the counters and prints them ------------------
func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) != 2 {
		log.Fatalf("usage: %s <feeds.tsv> <clean-rules.tsv>", os.Args[0])
	}
	cfgPath := args[0]
	rulesPath := args[1]

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

// Helper function to parse feeds with retry logic
func parseFeedWithRetry(fp *gofeed.Parser, name, url string, maxRetries int) (*gofeed.Feed, error) {
	var feed *gofeed.Feed
	var err error
	backoff := initialBackoff

	for attempt := 0; attempt <= maxRetries; attempt++ {
		feed, err = fp.ParseURL(url)
		if err == nil {
			return feed, nil
		}

		// Check if it's a 429 error
		if strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "Too Many Requests") {
			if attempt < maxRetries {
				log.Printf("  ⏳ rate limited on %s, waiting %v before retry %d/%d", name, backoff, attempt+1, maxRetries)
				time.Sleep(backoff)
				backoff *= 2 // exponential backoff
				continue
			}
		}

		// For non-429 errors or final attempt, return the error
		return nil, err
	}

	return nil, err
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
					// Use current date if published date is not available
					a.Published = time.Now()
				}
				out = append(out, a)
			}
			mu.Unlock()
			fmt.Printf("✅ %s\n", n)
		}(name, url)
	}
	wg.Wait()
	return out // always succeeds from the caller's point of view
}

// Helper function to fetch articles with retry logic
func fetchArticleWithRetry(url string, maxRetries int) (string, error) {
	var body string
	var err error
	backoff := initialBackoff

	for attempt := 0; attempt <= maxRetries; attempt++ {
		body, err = fetchArticle(url)
		if err == nil {
			return body, nil
		}

		// Check if it's a 429 error
		if strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "Too Many Requests") {
			if attempt < maxRetries {
				time.Sleep(backoff)
				backoff *= 2 // exponential backoff
				continue
			}
		}

		// For non-429 errors or final attempt, return the error
		return "", err
	}

	return "", err
}

// isDuplicateKeyError checks if an error is a MongoDB duplicate key error (E11000)
func isDuplicateKeyError(err error) bool {
	if err == nil {
		return false
	}
	// Check for MongoDB duplicate key error code
	if mongo.IsDuplicateKeyError(err) {
		return true
	}
	// Also check error message as fallback
	return strings.Contains(err.Error(), "E11000") || strings.Contains(err.Error(), "duplicate key")
}

func storeArticles(ctx context.Context, coll *mongo.Collection, arts []Article) (int, int, int, error) {
	// Build list of all links to check
	links := make([]string, len(arts))
	for i, a := range arts {
		links[i] = a.Link
	}

	// Find existing links in one query
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

	// Process only new articles
	var added, errs, skipped int
	for _, a := range arts {
		if existing[a.Link] {
			skipped++
			continue // Skip existing article
		}

		// Fetch and process new article
		body, err := fetchArticleWithRetry(a.Link, maxRetries)
		if err != nil {
			msg := err.Error()
			a.FetchError = &msg
			errs++
		} else {
			raw := body
			article := cleanText(a.Source, body)

			// Check if article length is less than MINLINE
			if len(article) < MINLINE {
				// Skip this article entirely - don't insert it
				skipped++
				continue
			}

			a.Raw = &raw
			a.Article = &article
		}

		// Initialize tags to empty slice
		a.Tags = []string{}

		// Insert the article, handling duplicate key errors gracefully
		_, err = coll.InsertOne(ctx, a)
		if err != nil {
			if isDuplicateKeyError(err) {
				// This is a duplicate - just skip it and continue
				skipped++
				continue
			}
			// For other errors, return them
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

					// Check if article length is less than MINLINE
					if len(article) < MINLINE {
						// Mark as too short so it won't come up in queries
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
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "rss2mongo/1.0 (+https://example.com/bot)")

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
