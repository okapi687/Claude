"""検証レポートを「共通 → 論文1 → 論文2 → 計画」の順に組み直す（11/22、一度だけ使う変換スクリプト）
- 章番号：1章 測り方とモデル、2章 論文1（旧4章）、3章 論文2（旧3章、並べ替え）、4章 計画（旧7章）
- レポート自身の図は「図N」をやめ、ファイル名と同じ「fig17」のように呼ぶ（論文の図と混ざらないように）
- 節の参照（「4.7 節」「| 4.7・4.8 |」など）を新しい番号に置き換える
引数なし。研究/paper/ の下を書き換える（git で元に戻せる）
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
V = os.path.join(PAPER, "検証")

SEC = {f"4.{i}": f"2.{i}" for i in range(1, 16)}
SEC.update({"3.1": "3.1", "3.2": "3.2", "3.3": "3.3", "3.4": "3.4", "3.5": "3.5",
            "3.6": "3.7", "3.7": "3.11", "3.8": "3.12", "3.9": "3.8", "3.10": "3.9", "3.11": "3.10"})
SEC.update({f"7.{i}": f"4.{i}" for i in range(1, 7)})

NUM = r"[3-47]\.\d{1,2}"
CHAIN = rf"{NUM}(?:[・〜、]{NUM})*"


def map_chain(s):
    return re.sub(NUM, lambda m: SEC.get(m.group(0), m.group(0)), s)


def remap_refs(text):
    # 「レポート 4.4」「4.7 節」「4.3・4.4 節」「| 4.7・4.8 |」を1回の走査で置き換える
    # （何度も走査すると 3.6→3.7→3.11 のように二重に置き換わるため）
    pat = rf"(レポート ?)({CHAIN})|(?<![\d.=/])({CHAIN})(?= ?節)|(?<=\| )({CHAIN})(?= \|)"

    def rep(m):
        if m.group(1):
            return m.group(1) + map_chain(m.group(2))
        return map_chain(m.group(3) or m.group(4))
    text = re.sub(pat, rep, text)
    # 「（下の 7.5）」「7.3 の計画E2」、計画の見出し
    text = text.replace("（下の 7.5）", "（下の 4.5）").replace("7.3 の計画E2", "4.3 の計画E2")
    text = re.sub(r"(?m)^### 7\.(\d)", r"### 4.\1", text)
    return text


def split_sections(path):
    """'### 3.1 ...' / '### 4.1 ...' で区切る。返り値: (先頭, やったこと表, {番号: 本文})"""
    t = open(path, encoding="utf-8").read()
    head, rest = t.split("## やったことと結果の一覧", 1)
    table, body = rest.split("\n---\n", 1)
    parts = re.split(r"(?m)^(### \d\.\d+ .*)$", body)
    secs = {}
    for i in range(1, len(parts), 2):
        num = re.match(r"### (\d\.\d+)", parts[i]).group(1)
        secs[num] = parts[i] + parts[i + 1]
    intro = parts[0]
    return head, "## やったことと結果の一覧" + table, intro, secs


FIGMAP = {}  # 旧「図N」 → ファイル番号


def collect_figs(text):
    for m in re.finditer(r"!\[[^\]]*\]\(\.\./report_fig/fig(\d+)_[^)]*\)\s*\n\s*\n\*\*図(\d+)\*\*", text):
        FIGMAP[m.group(2)] = m.group(1)


def relabel_captions(text):
    def rep(m):
        return f"![{m.group(1)}](../report_fig/fig{m.group(2)}{m.group(3)})\n\n**fig{m.group(2)}**"
    return re.sub(r"!\[([^\]]*)\]\(\.\./report_fig/fig(\d+)(_[^)]*)\)\s*\n\s*\n\*\*図\d+\*\*", rep, text)


def renum_heading(sec_text):
    return re.sub(r"^### (\d\.\d+)", lambda m: "### " + SEC[m.group(1)], sec_text, count=1)


# 旧ファイルを読む
h2, t2, i2, s2 = split_sections(os.path.join(V, "02_論文2.md"))
h1, t1, i1, s1 = split_sections(os.path.join(V, "03_論文1.md"))
for s in list(s1.values()) + list(s2.values()):
    collect_figs(s)

# レポート自身の図への参照（本文中）を fig 番号に置き換える（手で確認したものだけ）
OWN = [("（図2、表1）", "（fig02、表1）"), ("（図20 右）", "（fig20 右）"), ("（図14 左のオレンジ線）", "（fig09 左のオレンジ線）"),
       ("結果（図7 左、", "結果（fig12 左、"), ("（図22 の青線）", "（fig22 の青線）"), ("図26 の曲線", "fig26 の曲線"),
       ("（図12）", "（fig07）"), ("図7 左：論文1", "左：論文1")]


def fix_own(text):
    for a, b in OWN:
        text = text.replace(a, b)
    return text


def conv(text):
    return fix_own(relabel_captions(remap_refs(text)))


NAV = ("- 章の番号：1章（測り方とモデル）は [01](01_測り方とモデル.md)、2章（論文1）は [02](02_論文1_整列度.md)・[03](03_論文1_密度と相関.md)、"
       "3章（論文2）は [04](04_論文2_図3.md)・[05](05_論文2_他の図.md)、4章（計画）は [06](06_計画.md)。\n"
       "- 11/22 に、論文1 ⇒ 論文2 の順に並べ直して番号を振り直した。古い番号との対応は入口の「番号の対応」にある。\n"
       "- このレポート自身の図は fig17 のようにファイル名の番号で呼ぶ。「図3」「図11」のように書いたものは、論文の図。\n")

# 論文1（2章）
p1a = [f"4.{i}" for i in range(1, 9)]
p1b = [f"4.{i}" for i in range(9, 16)]
head1 = ("# 論文1の検証（速度一致あり）その1：整列度と α の効き方\n\n[← 検証レポート（入口）](../検証レポート.md)\n\n"
         "Gunji et al., *Robust swarm of soldier crabs* の再現。2.1〜2.8 節。2.9 節から先は [03_論文1_密度と相関](03_論文1_密度と相関.md)。\n\n" + NAV + "\n---\n\n")
intro1 = conv(i1).replace("## 4. 論文1の検証", "## 2. 論文1の検証")
body = head1 + conv(t1) + "\n---\n" + intro1 + "".join(conv(renum_heading(s1[k])) for k in p1a)
open(os.path.join(V, "02_論文1_整列度.md"), "w", encoding="utf-8").write(body.rstrip() + "\n")
head1b = ("# 論文1の検証（速度一致あり）その2：密度・速度一致の読み方・スケールフリー相関\n\n[← 検証レポート（入口）](../検証レポート.md)\n\n"
          "2.9〜2.15 節。2.1〜2.8 節と「やったこと → 結果」の表は [02_論文1_整列度](02_論文1_整列度.md)。\n\n" + NAV + "\n---\n\n")
body = head1b + "".join(conv(renum_heading(s1[k])) for k in p1b)
open(os.path.join(V, "03_論文1_密度と相関.md"), "w", encoding="utf-8").write(body.rstrip() + "\n")

# 論文2（3章）：3.1〜3.5 と新しい 3.6 → 04、3.7〜3.12 → 05
head2 = ("# 論文2の検証（速度一致なし）その1：図3 と α・ω・Int\n\n[← 検証レポート（入口）](../検証レポート.md)\n\n"
         "Murakami et al. 2017（Sci. Rep. 7:46447）と Appendix S1 の再現。3.1〜3.6 節。3.7 節から先は [05_論文2_他の図](05_論文2_他の図.md)。\n\n" + NAV + "\n---\n\n")
body = head2 + conv(t2) + "\n---\n" + conv(i2) + "".join(conv(renum_heading(s2[k])) for k in ["3.1", "3.2", "3.3", "3.4", "3.5"])
body += "\n<!-- 3.6 節（図3b,c の時間と平均の取り方）はここに足す -->\n"
open(os.path.join(V, "04_論文2_図3.md"), "w", encoding="utf-8").write(body.rstrip() + "\n")
head2b = ("# 論文2の検証（速度一致なし）その2：図S2・2体（図4）・図S3、まとめ\n\n[← 検証レポート（入口）](../検証レポート.md)\n\n"
          "3.7〜3.12 節。3.1〜3.6 節と「やったこと → 結果」の表は [04_論文2_図3](04_論文2_図3.md)。\n\n" + NAV + "\n---\n\n")
body = head2b + "".join(conv(renum_heading(s2[k])) for k in ["3.6", "3.9", "3.10", "3.11", "3.7", "3.8"])
open(os.path.join(V, "05_論文2_他の図.md"), "w", encoding="utf-8").write(body.rstrip() + "\n")

# 01・計画・変更一覧・入口は参照の置き換えだけ（中身は手で直す）
for name in ["01_測り方とモデル.md", "04_計画.md"]:
    t = open(os.path.join(V, name), encoding="utf-8").read()
    t = conv(t).replace("## 7. 計画", "## 4. 計画")
    open(os.path.join(V, "06_計画.md" if name == "04_計画.md" else name), "w", encoding="utf-8").write(t)
for path in [os.path.join(PAPER, "変更一覧.md"), os.path.join(PAPER, "検証レポート.md")]:
    t = open(path, encoding="utf-8").read()
    open(path, "w", encoding="utf-8").write(conv(t))
for old in ["02_論文2.md", "03_論文1.md", "04_計画.md"]:
    os.remove(os.path.join(V, old))
print("fig map (old 図N -> file):", dict(sorted(FIGMAP.items(), key=lambda x: int(x[0]))))
