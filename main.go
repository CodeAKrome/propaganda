package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
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
	Article     *string   `bson:"article,omitempty"`
	FetchError  *string   `bson:"fetch_error,omitempty"`
}

type Stats struct {
	ID           string         `bson:"_id"`
	SourceCounts map[string]int `bson:"source_counts"`
	Updated      time.Time      `bson:"updated"`
}

func main() {
	if len(os.Args) != 2 {
		log.Fatalf("usage: %s <config.tsv>", os.Args[0])
	}
	cfgPath := os.Args[1]

	ctx := context.Background()
	// client, err := mongo.Connect(ctx, options.Client().ApplyURI("mongodb://localhost:27017"))
	client, err := mongo.Connect(ctx, options.Client().ApplyURI("mongodb://"+os.Getenv("MONGO_USER")+":"+os.Getenv("MONGO_PASS")+"@localhost:27017"))
	if err != nil {
		log.Fatalf("mongo connect: %v", err)
	}
	defer client.Disconnect(ctx)

	articlesColl := client.Database(dbName).Collection(collName)
	statsColl := client.Database(dbName).Collection(statsCollName)

	// 1. read configuration file
	sources, err := readConfig(cfgPath)
	if err != nil {
		log.Fatalf("read config: %v", err)
	}

	// 2. read RSS feeds
	feeds, err := fetchAllFeeds(sources)
	if err != nil {
		log.Fatalf("fetch feeds: %v", err)
	}

	// 3. store in MongoDB
	if err := storeArticles(ctx, articlesColl, feeds); err != nil {
		log.Fatalf("store articles: %v", err)
	}

	// 4 & 5. backfill missing article text
	if err := backfillArticles(ctx, articlesColl); err != nil {
		log.Fatalf("backfill: %v", err)
	}

	// 6. update stats document
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

func fetchAllFeeds(src map[string]string) ([]Article, error) {
	var (
		mu   sync.Mutex
		out  []Article
		wg   sync.WaitGroup
		errs []error
	)

	for name, url := range src {
		wg.Add(1)
		go func(n, u string) {
			defer wg.Done()
			fp := gofeed.NewParser()
			feed, err := fp.ParseURL(u)
			if err != nil {
				mu.Lock()
				errs = append(errs, fmt.Errorf("%s: %w", n, err))
				mu.Unlock()
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

			// 🎯 announce completion
			fmt.Printf("✅ %s\n", n)
		}(name, url)
	}
	wg.Wait()

	if len(errs) > 0 {
		return out, fmt.Errorf("some feeds failed: %v", errs)
	}
	return out, nil
}

func storeArticles(ctx context.Context, coll *mongo.Collection, arts []Article) error {
	for _, a := range arts {
		// try to insert; if link or title already exists, mongo will raise E11000
		_, err := coll.InsertOne(ctx, a)
		if mongo.IsDuplicateKeyError(err) {
			continue // title or link already in table → ignore
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func backfillArticles(ctx context.Context, coll *mongo.Collection) error {
	cur, err := coll.Find(ctx, bson.M{"article": bson.M{"$exists": false}})
	if err != nil {
		return err
	}
	defer cur.Close(ctx)

	type job struct {
		id  interface{}
		url string
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
					update = bson.M{"$set": bson.M{"article": body, "fetch_error": nil}}
				}
				_, _ = coll.UpdateOne(ctx, bson.M{"_id": j.id}, update)
			}
		}()
	}

	for cur.Next(ctx) {
		var doc bson.M
		if err := cur.Decode(&doc); err != nil {
			continue
		}
		link, _ := doc["link"].(string)
		if link == "" {
			continue
		}
		jobs <- job{id: doc["_id"], url: link}
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
