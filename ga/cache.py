"""基因组 -> 评估结果的持久化缓存（SQLite）。

缓存键用**规范化后的基因组哈希**（见 genome.repair），因为不同基因型常常渲染出
同一份代码 —— 实测 div_mode=1 时 nr_steps 的 4 个取值完全等价，不去重就白烧 4 倍预算。
"""

import json
import sqlite3
import time


class Cache:
    def __init__(self, path="ga/results.sqlite"):
        self.con = sqlite3.connect(path)
        self.con.execute("""CREATE TABLE IF NOT EXISTS evals(
            key TEXT PRIMARY KEY, genome TEXT, ok INTEGER, tag TEXT,
            fitness REAL, v1_us REAL, headroom REAL, cv REAL,
            wall_s REAL, ts REAL, algo TEXT, gen INTEGER)""")
        self.con.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        r = self.con.execute(
            "SELECT ok,tag,fitness,v1_us,headroom,cv FROM evals WHERE key=?",
            (key,)).fetchone()
        if r is None:
            self.misses += 1
            return None
        self.hits += 1
        return dict(ok=bool(r[0]), tag=r[1], fitness=r[2],
                    v1_us=r[3], headroom=r[4], cv=r[5], cached=True)

    def put(self, key, genome, res, wall_s, algo="", gen=-1):
        self.con.execute(
            "INSERT OR REPLACE INTO evals VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, json.dumps(genome, sort_keys=True), int(res["ok"]), res["tag"],
             res.get("fitness", 0.0), res.get("v1_us", 0.0),
             res.get("headroom", 0.0), res.get("cv", 0.0),
             wall_s, time.time(), algo, gen))
        self.con.commit()

    def best(self, n=10):
        return self.con.execute(
            "SELECT fitness,v1_us,headroom,genome FROM evals WHERE ok=1 "
            "ORDER BY fitness DESC LIMIT ?", (n,)).fetchall()

    def stats(self):
        tot = self.con.execute("SELECT COUNT(*) FROM evals").fetchone()[0]
        ok = self.con.execute("SELECT COUNT(*) FROM evals WHERE ok=1").fetchone()[0]
        by_tag = dict(self.con.execute(
            "SELECT tag,COUNT(*) FROM evals GROUP BY tag").fetchall())
        return dict(total=tot, ok=ok, by_tag=by_tag,
                    hits=self.hits, misses=self.misses)
