"""11/23 の先生への報告資料（PowerPoint）を作る。引数: 出力 pptx
- base_theme.pptx：ゼミ資料.thmx（PowerPoint の「木版活字」テーマ）を PowerPoint で当てた空のプレゼンテーション
- 各トピックは「問題 → やったこと → 結果」の順（3列のスライド）と、結果の詳しいスライドの2枚組
- 図は ../検証/report_fig と ./img から読む
"""
import os, sys, copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "検証", "report_fig")
IMG = os.path.join(HERE, "img")

INK = RGBColor(0x32, 0x32, 0x32)
SUB = RGBColor(0x4A, 0x4A, 0x4A)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
TAN = RGBColor(0xE3, 0xDE, 0xD1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0x9F, 0x29, 0x36)      # 問題
BLUE = RGBColor(0x1B, 0x58, 0x7C)     # やったこと
ORANGE = RGBColor(0xB8, 0x5A, 0x08)   # 結果（テーマのオレンジ F07F09 を濃くした。白文字が読める）
LABEL = {"問題": RED, "やったこと": BLUE, "結果": ORANGE}

prs = Presentation(os.path.join(HERE, "base_theme.pptx"))
L_TITLE, L_CONTENT, L_SECTION = prs.slide_layouts[0], prs.slide_layouts[1], prs.slide_layouts[2]


# ---------------------------------------------------------------- 共通の部品
def slide_number(slide):
    """レイアウトのスライド番号プレースホルダーを複製して、番号のフィールドを入れる"""
    for ph in slide.slide_layout.placeholders:
        if ph.placeholder_format.type == 13:  # SLIDE_NUMBER
            sp = copy.deepcopy(ph._element)
            slide.shapes._spTree.append(sp)
            txBody = sp.find(qn("p:txBody"))
            for p in txBody.findall(qn("a:p")):
                txBody.remove(p)
            p = etree.SubElement(txBody, qn("a:p"))
            fld = etree.SubElement(p, qn("a:fld"), id="{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}", type="slidenum")
            etree.SubElement(fld, qn("a:rPr"), lang="ja-JP")
            t = etree.SubElement(fld, qn("a:t")); t.text = "‹#›"
            return


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def text_box(slide, x, y, w, h, paras, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             space=4, bullet=False):
    """paras は文字列か、(文字列, {size, color, bold}) のリスト"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, item in enumerate(paras):
        txt, opt = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        if bullet or opt.get("bullet"):
            set_bullet(p)
        r = p.add_run(); r.text = txt
        r.font.size = Pt(opt.get("size", size)); r.font.bold = opt.get("bold", bold)
        r.font.color.rgb = opt.get("color", color)
    return tb


def set_bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Inches(0.18)))); pPr.set("indent", str(-Emu(Inches(0.18))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bu = etree.SubElement(pPr, qn("a:buChar")); bu.set("char", "・")


def box(slide, x, y, w, h, fill, radius=0.06, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


def no_caps(shape, size=None):
    """テーマのタイトルは全部大文字（α が A になる）なので、大文字にしない"""
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            r.font._rPr.set("cap", "none")
            if size:
                r.font.size = Pt(size)


def content_slide(title, title_size=24):
    s = prs.slides.add_slide(L_CONTENT)
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx == 1:
            ph._element.getparent().remove(ph._element)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.28), Inches(8.8), Inches(0.75)
    t.text = title
    no_caps(t, title_size)
    t.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    slide_number(s)
    return s


def picture(slide, path, x, y, w, h):
    """箱 (x, y, w, h) に、縦横比を保って中央に入れる"""
    from PIL import Image
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    return slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def three_steps(slide, steps, y=1.2, h=3.75):
    """問題 → やったこと → 結果 の3列。steps は [(ラベル, [段落...]), ...]"""
    gap, x0, W = 0.25, 0.6, 8.8
    cw = (W - 2 * gap) / 3
    for i, (label, paras) in enumerate(steps):
        x = x0 + i * (cw + gap)
        chip = box(slide, x, y, cw, 0.42, LABEL[label], radius=0.25)
        tf = chip.text_frame; tf.margin_left = Inches(0.12)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = f"{i + 1}　{label}"; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = WHITE
        box(slide, x, y + 0.52, cw, h - 0.52, TAN, radius=0.04)
        text_box(slide, x + 0.1, y + 0.62, cw - 0.2, h - 0.72, paras, size=14, bullet=True, space=6)
        if i < 2:
            arr = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + cw + 0.04), Inches(y + 0.1),
                                         Inches(gap - 0.08), Inches(0.22))
            arr.fill.solid(); arr.fill.fore_color.rgb = MUTED; arr.line.fill.background()


def table(slide, x, y, w, rows, col_w=None, size=13, header_fill=INK, hl=None):
    """rows の1行目が見出し。hl は太字・赤にするセル {(行, 列)}"""
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(0.34 * nr)).table
    if col_w:
        for j, cw in enumerate(col_w):
            gt.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = gt.cell(i, j); c.text = ""
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03)
            p = c.text_frame.paragraphs[0]
            r = p.add_run(); r.text = val; r.font.size = Pt(size)
            if i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = header_fill
                r.font.color.rgb = WHITE; r.font.bold = True
            else:
                c.fill.solid(); c.fill.fore_color.rgb = WHITE if i % 2 else TAN
                r.font.color.rgb = INK
                if hl and (i, j) in hl:
                    r.font.bold = True; r.font.color.rgb = RED
    return gt


def takeaway(slide, text, y=4.62):
    b = box(slide, 0.6, y, 8.8, 0.46, INK, radius=0.12)
    tf = b.text_frame; tf.margin_left = Inches(0.15); tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = WHITE


def section(title, sub):
    s = prs.slides.add_slide(L_SECTION)
    s.shapes.title.text = title
    no_caps(s.shapes.title, 44)
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            ph.text = sub
    return s


# ---------------------------------------------------------------- 1 表紙
s = prs.slides.add_slide(L_TITLE)
s.shapes.title.text = "ミナミコメツキガニの群れの\n相互予期モデルの再現検証"
no_caps(s.shapes.title, 36)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 1:
        ph.text = "進捗報告（2026/11/23）\n電気通信大学 情報理工学域 一類 経営社会情報学プログラム　片岡 直己"
        for p in ph.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(14)
notes(s, "今週は、先週お送りした5つの課題に取り組みました。それぞれ、どんな問題があって、何を試して、結果はどうだったかの順に報告します。")

# ---------------------------------------------------------------- 2 前回まで
s = content_slide("前回までに行ったこと")
for i, (head, body) in enumerate([
        ("モデル検証", ["論文1（速度一致あり）と論文2（Murakami 2017、速度一致なし）を、別のモデルとして実装した",
                     "2つの最新版を MAM_basic_v1 にまとめ、論文の図と比べて、再現できていない点を洗い出した"]),
        ("残っていた主な問題", ["論文2：α による違い（赤はそろい、緑はそろわない）が出ない",
                          "論文1：α を大きくしても、整列度が下がりきらない"])]):
    x = 0.6 + i * 4.5
    box(s, x, 1.25, 4.3, 3.55, TAN, radius=0.04)
    text_box(s, x + 0.2, 1.4, 3.9, 0.45, [head], size=18, bold=True)
    text_box(s, x + 0.2, 1.95, 3.9, 2.7, body, size=15, bullet=True, space=10)
notes(s, "前回までに、論文1と論文2を別のモデルとして実装して、論文の図と比べました。一番大きな問題は、両方の論文で α の効き方が論文と違うことでした。")

# ---------------------------------------------------------------- 3 今週やったこと
s = content_slide("今週やったこと（先週お送りした5項目）")
table(s, 0.6, 1.3, 8.8, [
    ["", "問題", "やったこと"],
    ["論文2 ①", "floor と round のどちらがいいか", "遷移ベクトルの偏りを見た"],
    ["論文2 ②", "α を大きくすると整列度が上がった（論文と逆）", "α を細かく振って確かめた"],
    ["論文2 ③", "赤がそろわず、緑がそろった", "α の範囲を変えてみた"],
    ["論文2 ④", "速度の更新式（式と説明文が逆かもしれない）", "2つの読み方を比べた"],
    ["論文1", "α を大きくしても整列度が下がりきらない", "論文と実装の違いを1つずつ探した"],
], col_w=[1.2, 4.2, 3.4], size=15)
text_box(s, 0.6, 4.25, 8.8, 0.6, ["このあと、各項目を「問題 → やったこと → 結果」の順に説明します。"], size=14, color=SUB)
notes(s, "先週チャットでお送りした5項目です。")

# ---------------------------------------------------------------- 論文2
section("論文2", "速度一致なし（Murakami et al. 2017, Sci. Rep. 7:46447）")

# ① floor / round
s = content_slide("論文2 ①　floor と round のどちらがいいか")
three_steps(s, [
    ("問題", ["付録の Int は floor（X ≤ x < X+1）",
             "floor だと、青（本来ばらばら）まで整列度 0.99 になり、50 ステップでそろってしまう",
             "round にすると直るが、どちらが正しいか"]),
    ("やったこと", ["可能的遷移を整数化したときの遷移ベクトルの平均（偏り）を、floor と round で比べた",
                  "群れがそろうまでの時間も比べた"]),
    ("結果", ["floor は平均 (−0.51, −0.50) だけ左下にずれる。round はずれない",
             "基本ベクトルは自分の移動の向きで更新されるので、ずれが積み重なって全員が左下へ流れる",
             "round の方が論文に合う（次のスライド）"]),
])
notes(s, "論文2では、基本ベクトルの向きを自分が実際に動いた向きで更新します。そのため、floor の左下へのずれが毎ステップ積み重なり、全員がそろって左下へ流れてしまいます。")

s = content_slide("① 結果：round なら偏りがなく、そろう時間も論文に近い")
picture(s, os.path.join(FIG, "fig01_int_bias.png"), 0.6, 1.1, 3.9, 2.0)
text_box(s, 0.6, 3.1, 3.9, 0.3, ["遷移ベクトルの分布（floor は左下へずれる）"], size=11, color=MUTED)
table(s, 0.6, 3.42, 3.9, [["", "floor", "round", "論文"],
                          ["青の整列度", "0.99", "0.10", "0.2"],
                          ["赤がそろう時刻", "約60", "181〜1292", "約600"]],
      col_w=[1.35, 0.75, 1.05, 0.75], size=11, hl={(1, 2), (2, 2)})
picture(s, os.path.join(FIG, "fig29_planJ_p2_time.png"), 4.65, 1.1, 4.75, 2.75)
text_box(s, 4.65, 3.85, 4.75, 0.5, ["上：整列度、下：密度。左 α 2倍・round、中 α そのまま・round、右 α 2倍・floor（赤緑青は図3 の3条件）"], size=11, color=MUTED)
takeaway(s, "⇒ round の方が論文に合う。ただし付録の文面は floor（先生に確認したい）", y=4.6)
notes(s, "round にすると、青の整列度が 0.10 になり、図S2 の青のルールの割合（論文 63/5/32%）も 64/5/31% と合いました。赤がそろう時刻も、round ならシードごとに 181〜1292 ステップで、論文の約600 と同じくらいです。floor だと 60 ステップでそろってしまい速すぎます。")

# ② α を細かく
s = content_slide("論文2 ②　α を大きくすると整列度が上がった")
three_steps(s, [
    ("問題", ["Murakami 版（α そのまま・round）で、α を大きくすると整列度が上がった",
             "論文では、赤（α=0.2）がそろい、緑（α=0.5）はそろわない。逆になっている"]),
    ("やったこと", ["P=40 で、α を 0.05〜1.0 まで 0.05 刻みで細かく振った（3シード）",
                  "前回は α=0.1〜0.5 の5点しか見ていなかった"]),
    ("結果", ["整列度は単調に上がるのではなく、山の形になった",
             "山の頂上は α=0.4。論文（0.2〜0.3）のちょうど2倍の位置",
             "前回は、山の上り坂だけを見ていた"]),
])
notes(s, "α を広く振ると、α が大きいと整列度が下がる、という論文の振る舞い自体は出ました。ただし、山の位置が論文の2倍にずれています。")

s = content_slide("② 結果：整列度は山の形。頂上は論文の2倍の α")
picture(s, os.path.join(FIG, "fig14_paper2_alpha_sweep.png"), 0.6, 1.15, 8.8, 3.3)
takeaway(s, "⇒ 論文の赤・緑の値は、α を2倍した位置（白い菱形）に乗る", y=4.55)
notes(s, "黒い菱形が論文の赤と緑の値を α そのままの位置に置いたもの、白い菱形が2倍の位置に置いたものです。モデルの山は白い菱形の方に乗ります。α が小さいと、扇が狭くて他の個体と遷移先が重なりにくいので、なかなかそろいません。")

# ③ α の範囲
s = content_slide("論文2 ③　赤がそろわず、緑がそろった")
three_steps(s, [
    ("問題", ["赤（P=40, α=0.2）がそろわず、緑（P=40, α=0.5）がそろう",
             "②の山の位置が、論文の2倍にずれている"]),
    ("やったこと", ["可能的遷移の角度の範囲を ξ∈[−απ, απ] から ξ∈[−2απ, 2απ] に変えた",
                  "根拠：本文 p.4「α が大きいと、ほぼ全方向の近傍と相互作用する」。式どおりだと α=0.5 は前方 ±90° だけ"]),
    ("結果", ["赤 0.84・緑 0.13・青 0.12 で、図3 の順番（赤はそろい、緑・青はそろわない）が初めて出た",
             "図3a の格子の見た目も近い",
             "ただし式(1)・付録の式(3) には [−απ, απ] と明記"]),
])
notes(s, "2倍に読むのは、結果に合わせた読み方で、式の文面とは違います。ここは先生に確認したいところです。")

s = content_slide("③ 結果：α を2倍に読むと、図3 の順番と見た目が出る")
table(s, 0.6, 1.25, 3.95, [["条件", "論文", "そのまま", "2倍"],
                           ["青", "0.2", "0.11", "0.12"],
                           ["赤", "0.93", "0.22", "0.84"],
                           ["緑", "0.35", "0.78", "0.13"]],
      col_w=[0.8, 0.95, 1.1, 1.1], size=14, hl={(2, 3), (3, 3)})
text_box(s, 0.6, 2.75, 3.95, 1.6, [
    ("整列度（t=1000〜1500、ω 式どおり、round）", {"size": 11, "color": MUTED}),
    "赤は3シードとも t≈450〜1050 でそろう",
    "緑は論文より低め（0.13 と 0.35）"], size=14, bullet=False, space=6)
picture(s, os.path.join(FIG, "fig15_paper2_fig3a_grid.png"), 4.7, 1.2, 4.7, 3.25)
takeaway(s, "⇒ 図3 の順番・図3a の格子・図S3 は、α を2倍に読むと合う", y=4.55)
notes(s, "右の図は、論文の図3a と同じ P×α の格子です。左が α そのまま、右が2倍に読んだものです。2倍に読むと、α=0.2 で細長く進む群れ、α=0.5 で丸い塊になり、論文の図3a に近くなります。")

s = content_slide("③ 注意：図S2 の青のルールの割合は2倍だと合わない")
picture(s, os.path.join(FIG, "fig21_planD_blue.png"), 0.6, 1.15, 8.8, 3.25)
takeaway(s, "⇒ 青の相互予期 63% は α そのまま（±18°）でしか出ない（2倍だと 72%）", y=4.5)
notes(s, "相互予期の割合は、実際の扇の広さだけで決まります。図3 や図3a は2倍が合うのに、図S2 の青はそのままでないと合わない、という食い違いがあります。一定の倍率で読み替えるだけでは、両方を同時に合わせられません。")

# ④ 速度の更新式
s = content_slide("論文2 ④　速度（基本ベクトル）の更新式")
three_steps(s, [
    ("問題", ["式(13)：q = V_u + ω·v_u（ω は前の基本ベクトルの重み）",
             "図S3 の説明文は「ω が小さいと前の向きのまま」で、式と逆に読める",
             "V_u：直前の移動、v_u：前の基本ベクトル（どちらも単位ベクトル）"]),
    ("やったこと", ["2つの読み方を実装した",
                  "本文の2体のシミュレーション（図S2b）と、図S3（ω=0〜10）で比べた"]),
    ("結果", ["2体は、式どおりのときだけ論文と合う",
             "図S3 は、ω=0 だけ説明文どおりが合い、ω≥8 で下がる所は式どおりが合う",
             "2つの読み方は「ω を読むか 1/ω を読むか」と同じ"]),
])
notes(s, "向きだけを使うので、q = v_u + ω V_u と q = V_u + (1/ω) v_u は同じ向きになります。ω をどちら側に掛けるかは、ω を逆数にするかどうかと同じでした。")

s = content_slide("④ 結果：多くは式どおりが合う。ω=0 だけ説明文どおり")
text_box(s, 0.6, 1.15, 4.3, 0.35, ["2体のシミュレーション（図S2b、P=40、α=0.2）"], size=13, bold=True)
table(s, 0.6, 1.5, 4.3, [["", "式どおり", "説明文どおり", "論文"],
                         ["整列度", "0.93", "0.72", "約 0.9"],
                         ["相互予期", "0.83", "0.25", "0.88"],
                         ["自由移動", "0.17", "0.75", "0.12"]],
      col_w=[1.0, 1.0, 1.25, 1.05], size=13, hl={(1, 1), (2, 1), (3, 1)})
text_box(s, 5.1, 1.15, 4.3, 0.35, ["図S3（ω を変えたときの整列度、α 2倍）"], size=13, bold=True)
table(s, 5.1, 1.5, 4.3, [["ω", "0", "1", "5", "8"],
                         ["論文", "0.10", "0.70", "0.95", "0.55"],
                         ["式どおり", "0.65", "0.81", "0.83", "0.60"],
                         ["説明文どおり", "0.12", "0.81", "0.73", "0.72"]],
      col_w=[1.3, 0.75, 0.75, 0.75, 0.75], size=13, hl={(3, 1), (2, 4)})
text_box(s, 0.6, 3.05, 8.8, 1.4, [
    "説明文どおりだと、2体が離れてしまい、自由移動が 75% になる",
    "ω=0 のとき、式どおりでは θ が直前の移動の向きになり、移動を通してそろう（0.65）", "説明文どおりでは θ が最初のランダムな向きのまま変わらない（0.12）",
    "本文 p.4 は式どおり、図S3 の説明文は逆で、論文の中で食い違っている"], size=14, bullet=True, space=6)
takeaway(s, "⇒ 多くの図は式どおりに合う。ω=0 だけは説明文どおりでないと出ない", y=4.55)
notes(s, "今の v1 の既定は説明文どおりの読み方にしていましたが、式どおりの方が多くの図に合うので、見直したいと考えています。")

# ---------------------------------------------------------------- 論文1
section("論文1", "速度一致あり（Gunji et al., Robust swarm of soldier crabs）")

s = content_slide("論文1　α を大きくしても整列度が下がりきらない")
three_steps(s, [
    ("問題", ["α=0.3〜0.6 で、論文は 0.53 → 0.10 と下がる",
             "v1 は 0.58 → 0.37 で、下がりきらない",
             "α=0.3 → 0.4 の急な落ち方も出ない"]),
    ("やったこと", ["論文の文面と実装の違いになりそうな所を挙げ、1つずつ変えて試した",
                  "更新の順番、追従近傍の半径、整列度の定義、外部摂動、競合を文字どおりにする、α 2倍、速度一致で平均するもの など"]),
    ("結果", ["原因は、速度一致で全員の向き θ が完全にそろうこと（そろい具合 1.00）",
             "そのため整列度が扇の広さだけで決まり、α に対して一直線に下がる",
             "効いたのは2つ：α 2倍（候補A）と、速度一致で実際の移動を平均（候補B）"]),
])
notes(s, "式(8) の速度一致は近傍の θ を平均するだけなので、θ にばらつきが入る経路がありません。外部摂動を入れても、整列度が全体に下がるだけで、α の傾きは強くなりませんでした。")

s = content_slide("論文1 結果：2つの候補。論文の値はちょうどその間")
table(s, 0.6, 1.2, 8.8, [["設定", "α=0.3", "0.4", "0.5", "0.6", "θ のそろい"],
                         ["論文（図11）", "0.53", "0.30", "0.13", "0.10", "—"],
                         ["v1（α そのまま・floor）", "0.58", "0.49", "0.46", "0.37", "1.00"],
                         ["候補A：α 2倍・round", "0.37", "0.22", "0.10", "0.06", "1.00"],
                         ["候補B：round・移動で速度一致", "0.51", "0.43", "0.34", "0.25", "0.91 → 0.67"]],
      col_w=[3.0, 1.0, 1.0, 1.0, 1.0, 1.8], size=14, hl={(3, 3), (3, 4), (4, 1)})
text_box(s, 0.6, 3.05, 8.8, 1.4, [
    "候補A：α が大きい側が合う。ただし全員の向きが完全に同じで、整列度が時間で変わらない（動きが固すぎる）",
    "候補B：式(8) の言葉「近傍の平均の速度の向き」を、実際の移動の平均と読んだもの。θ がばらつき、α=0.3 が合う。α が大きい側は高すぎ、群れが分かれやすい",
    "どちらも round が必要（floor だと整列度が 0.18 より下がらない）"], size=14, bullet=True, space=6)
takeaway(s, "⇒ 速度一致で「何を平均するか」で、合う図が入れ替わる（先生に確認したい）", y=4.55)
notes(s, "論文1は1つに決まっていません。候補A は図11 の α が大きい側と密度、図8 の見た目に合い、候補B は α=0.3 と図13・14 の密度（Ls=10 と読んだとき）に合います。")

s = content_slide("論文1 結果：2つの候補の群れの様子")
picture(s, os.path.join(IMG, "fig30_paper1.png"), 0.6, 1.1, 6.0, 3.95)
text_box(s, 6.75, 1.25, 2.65, 3.7, [
    ("上：候補A", {"bold": True, "size": 15}),
    "1つの密な塊。向きは完全にそろう。整列度は時間でほとんど揺れない",
    ("下：候補B", {"bold": True, "size": 15}),
    "群れがいくつかに分かれ、それぞれ別の向きに進む。整列度は大きく揺れ続ける",
    ("右の列は整列度の時系列（5シード、橙 α=0.3、紫 α=0.5）", {"size": 11, "color": MUTED})], size=14, space=8)
notes(s, "論文の図14 では整列度が時間で広く分布するので、候補A の固すぎる動きは論文と違います。候補B は揺れますが、群れがまとまりきりません。")

# ---------------------------------------------------------------- そのほか
s = content_slide("そのほかに分かったこと")
cards = [("論文2 図4：2体の旋回の力", "論文の Methods どおり（向きの差は 0〜45°）に計算すると、どの設定でも論文と同じ形。前に合わなかったのは計算方法の違い"),
         ("論文1 の密度の測り方", "文面どおり空間を区画に分けると、図11 は Ls=25（2×2）で合う。群れが崩れたときの値 0.31 も一致"),
         ("論文1 図15：スケールフリー相関", "速度を基本ベクトル（論文は velocity と呼ぶ）で取ると、傾き −1.06 で論文（−0.96）と合う。実際の移動で取ると 1/10"),
         ("論文1 の相互予期 99%", "P≥10 では式(4) の決まりから自然に 98〜99% になる。論文2 の図S2 とも合い、問題ではない")]
for k, (h, b) in enumerate(cards):
    x = 0.6 + (k % 2) * 4.5; y = 1.2 + (k // 2) * 1.85
    box(s, x, y, 4.3, 1.7, TAN, radius=0.05)
    text_box(s, x + 0.15, y + 0.1, 4.0, 0.4, [h], size=15, bold=True, color=BLUE)
    text_box(s, x + 0.15, y + 0.5, 4.0, 1.15, [b], size=13, space=2)
notes(s, "先週の5項目以外に、検証の途中で分かったことです。詳しくは検証レポートにまとめています。")

# ---------------------------------------------------------------- まとめ
s = content_slide("まとめ")
for i, (head, col, body) in enumerate([
        ("できていること", BLUE, ["論文2：α 2倍・ω 式どおり・round で、図3 の順番・図3a・図S2 の赤緑・2体・図4・図S3（ω≥2）が合う",
                            "論文1：P=2 で群れが崩れる、P=1 で全員そろって直進する、図11 の密度（区画 Ls=25）"]),
        ("考えなければいけないこと", RED, ["α を2倍に読んでよいか（文面は [−απ, απ]）",
                                   "Int は floor か round か（文面は floor）",
                                   "論文1 の速度一致で何を平均するか（候補A か B か）",
                                   "論文2 の ω の掛け方（図S3 の ω=0 だけ合わない）"])]):
    x = 0.6 + i * 4.5
    box(s, x, 1.2, 4.3, 3.75, TAN, radius=0.04)
    text_box(s, x + 0.2, 1.3, 3.9, 0.45, [head], size=18, bold=True, color=col)
    text_box(s, x + 0.2, 1.85, 3.9, 3.0, body, size=14, bullet=True, space=8)
notes(s, "文面と違う読み方は2つ（α 2倍、round）です。どちらも合う方を選んだ結果なので、それで図が合っても、論文どおりに実装できた証拠にはなりません。")

s = content_slide("先生に確認したいこと")
qs = [("α の範囲", "式は [−απ, απ]。2倍に読むと多くの図に合うが、図S2 の青は合わなくなる"),
      ("Int", "文面は floor。論文2 は round でないと合わず、論文1 も floor では整列度が 0.18 より下がらない"),
      ("論文2 の ω の掛け方", "式どおりが多くの図に合うが、図S3 の ω=0 だけは説明文どおり"),
      ("論文1 の速度一致", "式(8) で、近傍の基本ベクトルの向きを平均するか、実際の移動を平均するか"),
      ("「再現できた」の基準", "論文の図は1本のシミュレーションの値。シードの範囲に入るかで判断してよいか")]
for k, (h, b) in enumerate(qs):
    y = 1.2 + k * 0.74
    c = box(s, 0.6, y, 0.55, 0.55, ORANGE, radius=0.5)
    p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = str(k + 1); r.font.size = Pt(18); r.font.bold = True; r.font.color.rgb = WHITE
    text_box(s, 1.3, y - 0.02, 2.6, 0.6, [h], size=15, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text_box(s, 3.9, y - 0.02, 5.5, 0.6, [b], size=13, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "このほかに、論文1 の密度の区画の大きさ Ls、図11 と図14 のどちらに合わせるか、スケールフリー相関の速度の取り方と規格化も確認したいです（検証レポートの 0.5 節）。")

# ---------------------------------------------------------------- 終わりに
section("終わりに", "今後の方針")

s = content_slide("今後の方針")
for k, (h, b) in enumerate([
        ("モデルのまとめ", "先生に確認したことをもとに既定の設定を決め、論文1・論文2 それぞれ別のファイルで最終版のコードを作る"),
        ("時計の完成と体長の大きい個体の実装", "モデルの検証が終わってから取りかかる。時計は論文1 の砂時計の設定（式(10a)(10b)）")]):
    y = 1.3 + k * 1.75
    c = box(s, 0.6, y, 0.7, 0.7, ORANGE, radius=0.5)
    p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = str(k + 1); r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = WHITE
    text_box(s, 1.5, y - 0.05, 7.9, 0.5, [h], size=20, bold=True)
    text_box(s, 1.5, y + 0.5, 7.9, 1.0, [b], size=15)
notes(s, "本と論文でも書いてあることが違いそう（パラメータやモデルの処理など）。論文を第一に信用するでOK？（論文を優先する方針にしています。）体長の大きい個体の参考資料は、あとでいただけると助かります。")

prs.save(sys.argv[1])
print("saved", sys.argv[1], "slides:", len(prs.slides))
