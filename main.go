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
	// feeds, err := fetchAllFeeds(sources)
	// if err != nil {
	// 	log.Fatalf("fetch feeds: %v", err)
	// }

	feeds := fetchAllFeeds(sources)

	// 4. store articles (raw + cleaned)  +  print per-source summary
	type sum struct{ added, errs int }
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
		add, errn, err := storeArticles(ctx, articlesColl, arts)
		if err != nil {
			log.Fatalf("store articles for %s: %v", src, err)
		}
		perSource[src] = sum{added: add, errs: errn}
	}

	for src, s := range perSource {
		fmt.Printf("✅ %-20s  added=%-4d  fetch-errors=%d\n", src, s.added, s.errs)
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

// func fetchAllFeeds(src map[string]string) ([]Article, error) {
// 	var (
// 		mu   sync.Mutex
// 		out  []Article
// 		wg   sync.WaitGroup
// 		errs []error
// 	)

// 	for name, url := range src {
// 		wg.Add(1)
// 		go func(n, u string) {
// 			defer wg.Done()
// 			fp := gofeed.NewParser()
// 			feed, err := fp.ParseURL(u)
// 			if err != nil {
// 				mu.Lock()
// 				errs = append(errs, fmt.Errorf("%s: %w", n, err))
// 				mu.Unlock()
// 				return
// 			}
// 			mu.Lock()
// 			for _, item := range feed.Items {
// 				a := Article{
// 					Source:      n,
// 					Title:       item.Title,
// 					Description: item.Description,
// 					Link:        item.Link,
// 				}
// 				if item.PublishedParsed != nil {
// 					a.Published = *item.PublishedParsed
// 				}
// 				out = append(out, a)
// 			}
// 			mu.Unlock()
// 			fmt.Printf("✅ %s\n", n)
// 		}(name, url)
// 	}
// 	wg.Wait()

// 	if len(errs) > 0 {
// 		return out, fmt.Errorf("some feeds failed: %v", errs)
// 	}
// 	return out, nil
// }

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
			feed, err := fp.ParseURL(u)
			if err != nil {
				log.Printf("⚠️  feed failed: %s → %v", n, err) // keep calm and carry on
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
				}
				out = append(out, a)
			}
			mu.Unlock()
			fmt.Printf("✅ %s\n", n)
		}(name, url)
	}
	wg.Wait()
	return out // always succeeds from the caller’s point of view
}

// func storeArticles(ctx context.Context, coll *mongo.Collection, arts []Article) (int, int, error) {
// 	var added, errs int
// 	for _, a := range arts {
// 		body, err := fetchArticle(a.Link)
// 		var raw, article *string
// 		if err != nil {
// 			msg := err.Error()
// 			a.FetchError = &msg
// 			errs++
// 		} else {
// 			raw = &body
// 			cleaned := cleanText(a.Source, body)
// 			article = &cleaned
// 		}
// 		a.Raw = raw
// 		a.Article = article
// 		a.Tags = []string{} // <-- NEW: always init to []

// 		_, err = coll.InsertOne(ctx, a)
// 		if mongo.IsDuplicateKeyError(err) {
// 			continue // nothing inserted
// 		}
// 		if err != nil {
// 			return added, errs, err
// 		}
// 		added++
// 	}
// 	return added, errs, nil
// }

func storeArticles(ctx context.Context, coll *mongo.Collection, arts []Article) (int, int, error) {
	var added, errs int
	for _, a := range arts {
		body, err := fetchArticle(a.Link)
		var raw, article *string
		if err != nil {
			msg := err.Error()
			a.FetchError = &msg
			errs++
		} else {
			raw = &body
			cleaned := cleanText(a.Source, body)
			article = &cleaned
		}
		a.Raw = raw
		a.Article = article
		a.Tags = []string{} // always init to []

		// --- keep existing record if it is already there ---
		_, err = coll.ReplaceOne(
			ctx,
			bson.M{"link": a.Link},            // unique key
			a,                                 // new document
			options.Replace().SetUpsert(true), // insert if not found
		)
		if err != nil {
			return added, errs, err
		}
		added++ // we count it as “added” even if it was only an upsert
	}
	return added, errs, nil
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
				body, err := fetchArticle(j.url)
				var update bson.M
				if err != nil {
					msg := err.Error()
					update = bson.M{"$set": bson.M{"fetch_error": msg}}
				} else {
					raw := body
					article := cleanText(j.source, body)
					update = bson.M{"$set": bson.M{
						"raw":         raw,
						"article":     article,
						"fetch_error": nil,
					}}
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
