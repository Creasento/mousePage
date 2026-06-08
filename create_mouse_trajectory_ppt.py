import html
import os
import zipfile
from pathlib import Path


EMU_PER_INCH = 914400
SLIDE_W = 13.333
SLIDE_H = 7.5
SLIDE_W_EMU = int(SLIDE_W * EMU_PER_INCH)
SLIDE_H_EMU = int(SLIDE_H * EMU_PER_INCH)


OUT = Path("mouse_trajectory_model_presentation.pptx")


def emu(value):
    return int(value * EMU_PER_INCH)


def esc(text):
    return html.escape(str(text), quote=True)


def text_runs(lines, font_size=18, bold=False, color="222222"):
    if isinstance(lines, str):
        lines = [lines]
    body = []
    for idx, line in enumerate(lines):
        bullet = False
        text = line
        if text.startswith("- "):
            bullet = True
            text = text[2:]
        mar = ' marL="285750" indent="-171450"' if bullet else ""
        bu = '<a:buChar char="•"/>' if bullet else '<a:buNone/>'
        end_para = "" if idx == len(lines) - 1 else "<a:endParaRPr/>"
        body.append(
            f"""
            <a:p>
              <a:pPr{mar}>{bu}</a:pPr>
              <a:r>
                <a:rPr lang="en-US" sz="{font_size * 100}" b="{1 if bold else 0}">
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                </a:rPr>
                <a:t>{esc(text)}</a:t>
              </a:r>
              {end_para}
            </a:p>
            """
        )
    return "\n".join(body)


def textbox(shape_id, x, y, w, h, lines, font_size=18, bold=False, color="222222"):
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" anchor="t"/>
        <a:lstStyle/>
        {text_runs(lines, font_size, bold, color)}
      </p:txBody>
    </p:sp>
    """


def rect(shape_id, x, y, w, h, fill="F5F7FA", line="D9DEE7"):
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="Rect {shape_id}"/>
        <p:cNvSpPr/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
      </p:spPr>
    </p:sp>
    """


def title(text):
    return textbox(2, 0.55, 0.32, 12.2, 0.55, text, 28, True, "111111")


def subtitle(text):
    return textbox(3, 0.58, 0.92, 12.1, 0.35, text, 13, False, "555555")


def image_pic(shape_id, r_id, x, y, w, h):
    return f"""
    <p:pic>
      <p:nvPicPr>
        <p:cNvPr id="{shape_id}" name="Picture {shape_id}"/>
        <p:cNvPicPr/>
        <p:nvPr/>
      </p:nvPicPr>
      <p:blipFill>
        <a:blip r:embed="{r_id}"/>
        <a:stretch><a:fillRect/></a:stretch>
      </p:blipFill>
      <p:spPr>
        <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
      </p:spPr>
    </p:pic>
    """


def table_like(shape_id_start, x, y, col_w, row_h, rows, header_fill="EEF3F8"):
    shapes = []
    sid = shape_id_start
    for r, row in enumerate(rows):
        for c, txt in enumerate(row):
            fill = header_fill if r == 0 else "FFFFFF"
            shapes.append(rect(sid, x + sum(col_w[:c]), y + r * row_h, col_w[c], row_h, fill=fill))
            sid += 1
            size = 9 if len(str(txt)) > 18 else 10
            shapes.append(textbox(sid, x + sum(col_w[:c]) + 0.05, y + r * row_h + 0.04, col_w[c] - 0.1, row_h - 0.06, txt, size, r == 0))
            sid += 1
    return "\n".join(shapes)


def slide_xml(shapes):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>
      </p:grpSpPr>
      {shapes}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def slide_rels(image_targets):
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    for idx, target in enumerate(image_targets, start=2):
        rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>"""


def presentation_xml(n):
    ids = "\n".join([f'<p:sldId id="{255+i}" r:id="rId{i+2}"/>' for i in range(n)])
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W_EMU}" cy="{SLIDE_H_EMU}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(n):
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    ]
    for i in range(n):
        rels.append(
            f'<Relationship Id="rId{i+3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def content_types(n, media):
    slide_overrides = "".join([f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, n + 1)])
    png_default = '<Default Extension="png" ContentType="image/png"/>'
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {png_default}
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  {slide_overrides}
</Types>"""


def minimal_master():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
 <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
 <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>"""


def minimal_layout():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank">
 <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
</p:sldLayout>"""


def theme():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Clean">
 <a:themeElements>
  <a:clrScheme name="Clean"><a:dk1><a:srgbClr val="111111"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="222222"/></a:dk2><a:lt2><a:srgbClr val="F5F7FA"/></a:lt2><a:accent1><a:srgbClr val="2F6B9A"/></a:accent1><a:accent2><a:srgbClr val="D94F45"/></a:accent2><a:accent3><a:srgbClr val="5BA66B"/></a:accent3><a:accent4><a:srgbClr val="E0A72E"/></a:accent4><a:accent5><a:srgbClr val="7A5AA6"/></a:accent5><a:accent6><a:srgbClr val="4BA3A0"/></a:accent6><a:hlink><a:srgbClr val="2F6B9A"/></a:hlink><a:folHlink><a:srgbClr val="7A5AA6"/></a:folHlink></a:clrScheme>
  <a:fontScheme name="Clean"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme>
  <a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
 </a:themeElements>
</a:theme>"""


def add_slide(slides, shapes, images=None):
    slides.append((slide_xml(shapes), images or []))


def build_slides():
    slides = []
    img_evolution = "model_optimization_trajectory_evolution_with_real_yexpanded.png"
    img_heatmap = "evaluation_heatmap.png"
    img_best_grid = "trajectory_grid_temp0p5_points.png"

    add_slide(
        slides,
        title("Human-Like Mouse Trajectory Generation")
        + subtitle("Conditional generative modeling of Fitts-task cursor movements")
        + textbox(4, 0.85, 2.2, 6.1, 2.0, ["CVAE + GRU trajectory generator", "Conditioned on A, W, ID, duration", "Evaluated against original A/W task conditions"], 22, True)
        + textbox(5, 7.1, 2.25, 5.2, 1.8, ["Key result", "- Current best: weak late dynamics + peak velocity control", "- Mean gap: 0.156", "- Path/deviation close to real data"], 18),
    )

    add_slide(
        slides,
        title("Model: Conditional CVAE + GRU")
        + subtitle("Generate normalized trajectories under movement conditions")
        + textbox(4, 0.7, 1.55, 4.0, 4.1, ["Inputs", "- A: movement distance", "- W: target width", "- ID: difficulty", "- duration"], 20)
        + textbox(5, 4.75, 1.55, 3.8, 4.1, ["Model", "- Encoder: GRU", "- Latent z: trajectory variation", "- Decoder: GRU", "- Endpoint constrained"], 20)
        + textbox(6, 8.75, 1.55, 3.9, 4.1, ["Output", "- Normalized path", "- start=(0,0)", "- target=(1,0)", "- remapped to screen"], 20),
    )

    add_slide(
        slides,
        title("Why This Model")
        + textbox(4, 0.75, 1.45, 5.5, 4.7, ["CVAE", "- Same condition has many valid human paths", "- Latent space captures movement variation", "- Supports conditional sampling"], 21)
        + textbox(5, 6.85, 1.45, 5.5, 4.7, ["GRU Decoder", "- Trajectories are sequential", "- Generates time-ordered points", "- Lightweight and stable"], 21),
    )

    add_slide(
        slides,
        title("Data Repair and Normalization")
        + textbox(4, 0.75, 1.35, 5.8, 4.8, ["Issue", "- CSV lacked enough trial identity columns", "- Multiple original trajectories were merged", "- ID, duration, t_norm were missing"], 20)
        + textbox(5, 7.0, 1.35, 5.4, 4.8, ["Repair", "- Split when point_index resets", "- Added segment_index", "- Recomputed ID, duration, t_norm", "- Updated GROUP_KEYS"], 20)
        + textbox(6, 0.8, 6.25, 11.6, 0.45, "Result: 7,525,267 rows | 43,470 segments | 41,431 usable non-error trials", 18, True, "2F6B9A"),
    )

    add_slide(
        slides,
        title("Evaluation Setup")
        + textbox(4, 0.7, 1.35, 4.3, 4.8, ["Original Conditions", "- A ∈ {300, 301, 900, 901}", "- W ∈ {20, 50, 120}", "- 12 A/W combinations"], 20)
        + textbox(5, 4.9, 1.35, 4.0, 4.8, ["Generation", "- temperature 0.0–0.9", "- 20 samples per condition/temp", "- compare generated vs real"], 20)
        + textbox(6, 8.75, 1.35, 4.1, 4.8, ["Metrics", "- path length", "- max deviation", "- peak velocity", "- acceleration", "- jerk"], 20),
    )

    rows = [
        ["Model", "Main Change", "Outcome"],
        ["Baseline", "CVAE+GRU, seq64", "Too smooth"],
        ["Dynamic", "Pointwise vel/acc loss", "Dynamics still collapsed"],
        ["Statistic", "Acc/jerk distribution loss", "Dynamics recovered"],
        ["Strong late", "Late correction loss", "Peak velocity explosion"],
        ["Late+peak", "Weak late + peak control", "Current best"],
    ]
    add_slide(
        slides,
        title("Optimization History")
        + table_like(4, 0.65, 1.35, [2.2, 4.4, 5.4], 0.72, rows),
    )

    rows = [
        ["Model", "Gap ↓", "Path", "Dev.", "Peak", "Acc.", "Jerk"],
        ["Dynamic", "0.421", "0.97x", "0.79x", "1.22x", "0.24x", "0.05x"],
        ["Statistic", "0.233", "1.01x", "0.73x", "1.41x", "0.72x", "0.68x"],
        ["Strong late", "0.416", "1.02x", "0.97x", "2.24x", "0.88x", "0.75x"],
        ["Late+peak", "0.156", "1.03x", "0.97x", "1.18x", "0.79x", "0.69x"],
    ]
    add_slide(
        slides,
        title("Quantitative Results")
        + subtitle("Ratios are generated / real. Closer to 1.0x is better.")
        + table_like(4, 0.55, 1.45, [2.3, 1.15, 1.25, 1.25, 1.25, 1.25, 1.25], 0.75, rows)
        + textbox(99, 0.7, 6.0, 12.0, 0.5, "Late+peak gives the best overall balance: shape, deviation, and velocity are close to real data.", 17, True, "2F6B9A"),
    )

    add_slide(
        slides,
        title("Trajectory Evolution")
        + subtitle("Real data → baseline → optimized models")
        + image_pic(4, "rId2", 0.3, 1.35, 12.75, 5.6),
        [("generated/" + img_evolution, img_evolution)],
    )

    add_slide(
        slides,
        title("Current Best: Condition-Level Evaluation")
        + subtitle("Weak late dynamics + peak velocity control")
        + image_pic(4, "rId2", 0.75, 1.25, 11.8, 5.65),
        [("generated/original_conditions_seq128_late_peak/" + img_heatmap, img_heatmap)],
    )

    add_slide(
        slides,
        title("Current Best: Generated Trajectories")
        + subtitle("Temperature = 0.5, original A/W conditions")
        + image_pic(4, "rId2", 0.25, 1.15, 12.85, 5.9),
        [("generated/original_conditions_seq128_late_peak/" + img_best_grid, img_best_grid)],
    )

    add_slide(
        slides,
        title("Similarity to Real Data")
        + textbox(4, 0.75, 1.35, 5.8, 4.7, ["Well matched", "- Path length: 1.03x real", "- Max deviation: 0.97x real", "- Peak velocity: 1.18x real", "- Condition-dependent shapes"], 21)
        + textbox(5, 7.0, 1.35, 5.3, 4.7, ["Remaining gaps", "- Acceleration: 0.79x real", "- Jerk: 0.69x real", "- Weakest: short distance + wide target", "- Some loop-like samples remain"], 21),
    )

    add_slide(
        slides,
        title("Key Insights")
        + textbox(4, 0.8, 1.3, 11.8, 4.9, ["- Position loss alone is not enough for human-like trajectories", "- Pointwise dynamic losses can average out micro-adjustments", "- Distribution-level acceleration and jerk losses are more effective", "- Late correction must be weak and paired with peak velocity control", "- Current best balances trajectory shape and dynamics"], 23),
    )

    add_slide(
        slides,
        title("Applications and Expected Impact")
        + textbox(4, 0.75, 1.35, 5.8, 4.8, ["Applications", "- Human-like cursor simulation", "- HCI pointing task modeling", "- Synthetic trajectory generation", "- Adaptive UI / accessibility"], 21)
        + textbox(5, 7.0, 1.35, 5.3, 4.8, ["Expected Impact", "- Fewer real trials needed", "- Condition-aware movement synthesis", "- Richer interaction simulation", "- Better evaluation of target designs"], 21),
    )

    add_slide(
        slides,
        title("Conclusion")
        + textbox(4, 0.9, 1.45, 11.5, 4.9, ["Final model", "- CVAE+GRU with distribution-level dynamics losses", "- Best version uses weak late correction + peak velocity control", "- Mean gap reduced to 0.156", "- Realistic path shape and improved temporal dynamics"], 24)
        + textbox(5, 0.95, 6.25, 11.4, 0.45, "Next: residual decoder and condition-specific temperature/loss weighting", 18, True, "2F6B9A"),
    )

    return slides


def write_pptx():
    slides = build_slides()
    media_files = {}
    for _, images in slides:
        for source, target in images:
            media_files[target] = source

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides), media_files))
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>""")
        z.writestr("docProps/core.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Mouse Trajectory Generation Model</dc:title><dc:creator>Codex</dc:creator></cp:coreProperties>""")
        z.writestr("docProps/app.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application></Properties>""")
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", minimal_master())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>""")
        z.writestr("ppt/slideLayouts/slideLayout1.xml", minimal_layout())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""")
        z.writestr("ppt/theme/theme1.xml", theme())
        for i, (xml, images) in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", xml)
            targets = [target for _, target in images]
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels(targets))
        for target, source in media_files.items():
            z.write(source, f"ppt/media/{target}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    write_pptx()
