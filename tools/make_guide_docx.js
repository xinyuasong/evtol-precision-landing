// Build a professional Word version of docs/GUIDE.md.
//   node tools/make_guide_docx.js docs/GUIDE.md "out.docx"
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak, ImageRun,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  Header, Footer, PageNumber, LevelFormat, convertInchesToTwip,
} = require("docx");

const SRC = process.argv[2] || "docs/GUIDE.md";
const OUT = process.argv[3] || "eVTOL_Precision_Landing_Guide.docx";

const BODY = "Calibri";
const MONO = "Consolas";
const ACCENT = "1F3864";   // deep navy
const ACCENT2 = "2E74B5";  // mid blue
const CODEBG = "F4F5F7";
const RULE = "D0D4DA";
const MUTED = "595959";

// ---------- inline formatting: **bold**, `code`, *italic* ----------
function runs(text, opts = {}) {
  const base = { font: BODY, size: opts.size || 21, color: opts.color };
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    const tok = m[0];
    if (tok.startsWith("**")) {
      out.push(new TextRun({ ...base, text: tok.slice(2, -2), bold: true }));
    } else if (tok.startsWith("`")) {
      out.push(new TextRun({ ...base, font: MONO, size: (opts.size || 21) - 2, text: tok.slice(1, -1), color: "A3324C" }));
    } else {
      out.push(new TextRun({ ...base, text: tok.slice(1, -1), italics: true }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out.length ? out : [new TextRun({ ...base, text: "" })];
}

function para(text, opts = {}) {
  return new Paragraph({
    children: runs(text, opts),
    spacing: { after: opts.after ?? 140, line: 276 },
    alignment: opts.alignment,
    indent: opts.indent,
  });
}

function codeBlock(lines, lang) {
  const out = [];
  if (lang) {
    out.push(new Paragraph({
      children: [new TextRun({ text: lang.toUpperCase(), font: BODY, size: 14, color: MUTED, bold: true })],
      spacing: { before: 120, after: 0 },
      shading: { type: ShadingType.CLEAR, fill: CODEBG },
      indent: { left: 170, right: 170 },
      border: { top: { style: BorderStyle.SINGLE, size: 6, color: RULE } },
    }));
  }
  lines.forEach((l, i) => {
    out.push(new Paragraph({
      children: [new TextRun({ text: l || " ", font: MONO, size: 17, color: "1B1F24" })],
      spacing: { before: 0, after: i === lines.length - 1 ? 0 : 20, line: 240 },
      shading: { type: ShadingType.CLEAR, fill: CODEBG },
      indent: { left: 170, right: 170 },
    }));
  });
  out.push(new Paragraph({
    children: [new TextRun({ text: "", size: 4 })],
    spacing: { before: 0, after: 160 },
    shading: { type: ShadingType.CLEAR, fill: CODEBG },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE } },
  }));
  return out;
}

function splitRow(line) {
  return line.replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function table(rows) {
  const head = rows[0];
  const body = rows.slice(1);
  const total = convertInchesToTwip(6.5);
  const widths = head.length === 2 ? [Math.round(total * 0.3), Math.round(total * 0.7)]
    : Array(head.length).fill(Math.round(total / head.length));
  widths[widths.length - 1] = total - widths.slice(0, -1).reduce((a, b) => a + b, 0);

  const cell = (txt, i, isHead) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: isHead ? ACCENT : "FFFFFF" },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      children: isHead
        ? [new TextRun({ text: txt, font: BODY, size: 19, bold: true, color: "FFFFFF" })]
        : runs(txt, { size: 19 }),
      spacing: { after: 0, line: 240 },
    })],
  });

  return new Table({
    columnWidths: widths,
    width: { size: total, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      right: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      insideVertical: { style: BorderStyle.SINGLE, size: 4, color: RULE },
    },
    rows: [
      new TableRow({ tableHeader: true, children: head.map((c, i) => cell(c, i, true)) }),
      ...body.map((r) => new TableRow({ children: widths.map((_, i) => cell(r[i] ?? "", i, false)) })),
    ],
  });
}

// ---------- parse the markdown ----------
const md = fs.readFileSync(SRC, "utf8").split("\n");
const children = [];
const toc = [];   // {level, text} for the static contents list
let i = 0;
let seenFirstH1 = false;

while (i < md.length) {
  const line = md[i];

  // fenced code
  if (line.startsWith("```")) {
    const lang = line.slice(3).trim();
    const buf = [];
    i++;
    while (i < md.length && !md[i].startsWith("```")) buf.push(md[i++]);
    i++;
    children.push(...codeBlock(buf, lang));
    continue;
  }

  // table
  if (line.startsWith("|") && md[i + 1] && /^\|[-:| ]+\|$/.test(md[i + 1].replace(/\s/g, "").replace(/-+/g, "-").replace(/^\|/, "|"))) {
    const rows = [splitRow(line)];
    i += 2;
    while (i < md.length && md[i].startsWith("|")) rows.push(splitRow(md[i++]));
    children.push(table(rows));
    children.push(new Paragraph({ children: [new TextRun({ text: "", size: 10 })], spacing: { after: 180 } }));
    continue;
  }

  // headings
  if (line.startsWith("#")) {
    const level = line.match(/^#+/)[0].length;
    const text = line.replace(/^#+\s*/, "").replace(/[`*]/g, "");
    // the document's own title line is replaced by the title page
    if (level === 1 && !seenFirstH1) {
      seenFirstH1 = true;
      i++;
      while (i < md.length && md[i].trim() !== "") i++;   // drop its lead paragraph too
      continue;
    }
    if (level <= 2) toc.push({ level, text: text.replace(/[`*]/g, "") });
    if (level === 1) {
      children.push(new Paragraph({
        children: [new PageBreak()],
      }));
      children.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({ text, font: BODY, size: 34, bold: true, color: ACCENT })],
        spacing: { before: 0, after: 60 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: ACCENT2 } },
      }));
      children.push(new Paragraph({ children: [new TextRun({ text: "", size: 8 })], spacing: { after: 160 } }));
    } else if (level === 2) {
      children.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text, font: BODY, size: 26, bold: true, color: ACCENT2 })],
        spacing: { before: 320, after: 120 },
      }));
    } else {
      children.push(new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun({ text, font: BODY, size: 22, bold: true, color: "44546A" })],
        spacing: { before: 220, after: 100 },
      }));
    }
    i++;
    continue;
  }

  // image:  ![caption](path)
  const img = line.match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/);
  if (img) {
    const file = path.resolve(path.dirname(SRC), img[2]);
    if (fs.existsSync(file)) {
      const dim = require("child_process").execSync(
        `python3 -c "import cv2;im=cv2.imread('${file}');print(im.shape[1],im.shape[0])"`).toString().trim().split(" ").map(Number);
      const maxW = 600;
      const w = Math.min(maxW, dim[0]);
      const h = Math.round(dim[1] * (w / dim[0]));
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 80 },
        children: [new ImageRun({ type: path.extname(file).slice(1), data: fs.readFileSync(file), transformation: { width: w, height: h } })],
      }));
      if (img[1]) {
        children.push(new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [new TextRun({ text: img[1], font: BODY, size: 17, italics: true, color: MUTED })],
        }));
      }
    }
    i++;
    continue;
  }

  // horizontal rule -> skip (headings already carry a rule)
  if (/^-{3,}$/.test(line.trim())) { i++; continue; }

  // bullets
  if (/^[-*]\s+/.test(line)) {
    while (i < md.length && /^[-*]\s+/.test(md[i])) {
      children.push(new Paragraph({
        children: runs(md[i].replace(/^[-*]\s+/, "")),
        numbering: { reference: "guide-bullets", level: 0 },
        indent: { left: 460, hanging: 260 },
        spacing: { after: 80, line: 276 },
      }));
      i++;
    }
    continue;
  }

  // numbered list
  if (/^\d+\.\s+/.test(line)) {
    while (i < md.length && /^\d+\.\s+/.test(md[i])) {
      children.push(new Paragraph({
        children: runs(md[i].replace(/^\d+\.\s+/, "")),
        numbering: { reference: "guide-numbers", level: 0 },
        indent: { left: 500, hanging: 300 },
        spacing: { after: 80, line: 276 },
      }));
      i++;
    }
    continue;
  }

  if (line.trim() === "") { i++; continue; }

  // paragraph (join wrapped lines)
  const buf = [line];
  i++;
  while (i < md.length && md[i].trim() !== "" && !md[i].startsWith("#") && !md[i].startsWith("```")
         && !md[i].startsWith("|") && !/^[-*]\s+/.test(md[i]) && !/^\d+\.\s+/.test(md[i]) && !/^-{3,}$/.test(md[i].trim())) {
    buf.push(md[i++]);
  }
  children.push(para(buf.join(" ")));
}

// ---------- title page + TOC ----------
const today = new Date().toLocaleDateString("en-CA", { year: "numeric", month: "long", day: "numeric" });
const front = [
  new Paragraph({ children: [new TextRun({ text: "", size: 24 })], spacing: { after: 2400 } }),
  new Paragraph({
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: "Autonomous Vision-Guided Precision Landing", font: BODY, size: 52, bold: true, color: ACCENT })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "A simulation-validated flight control stack", font: BODY, size: 30, color: ACCENT2 })],
    spacing: { after: 60 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT2 } },
  }),
  new Paragraph({ children: [new TextRun({ text: "", size: 8 })], spacing: { after: 400 } }),
  new Paragraph({
    children: [new TextRun({ text: "Setup, operation, testing, and system reference", font: BODY, size: 24, color: MUTED })],
    spacing: { after: 1600 },
  }),
  new Paragraph({ children: [new TextRun({ text: "Richard Song", font: BODY, size: 24, bold: true })], spacing: { after: 40 } }),
  new Paragraph({ children: [new TextRun({ text: "Mechatronics Engineering, University of Waterloo", font: BODY, size: 21, color: MUTED })], spacing: { after: 40 } }),
  new Paragraph({ children: [new TextRun({ text: today, font: BODY, size: 21, color: MUTED })], spacing: { after: 600 } }),
  new Paragraph({
    children: [new TextRun({ text: "Status: validated in simulation. This stack has not flown.", font: BODY, size: 20, bold: true, color: "A3324C" })],
    shading: { type: ShadingType.CLEAR, fill: "FBEEF1" },
    spacing: { before: 200, after: 200 },
    indent: { left: 170, right: 170 },
  }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text: "Contents", font: BODY, size: 34, bold: true, color: ACCENT })],
    spacing: { after: 60 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: ACCENT2 } },
  }),
  new Paragraph({ children: [new TextRun({ text: "", size: 8 })], spacing: { after: 240 } }),
  ...toc.map((h) => new Paragraph({
    children: [new TextRun({
      text: h.text,
      font: BODY,
      size: h.level === 1 ? 23 : 21,
      bold: h.level === 1,
      color: h.level === 1 ? ACCENT : "333333",
    })],
    spacing: { before: h.level === 1 ? 200 : 0, after: h.level === 1 ? 60 : 60, line: 260 },
    indent: { left: h.level === 1 ? 0 : 340 },
  })),
];

// the first body element is a PageBreak paragraph before PART 1 — the TOC page already ends,
// so drop it to avoid a blank page
if (children.length && children[0].constructor === Paragraph) children.shift();

const doc = new Document({
  creator: "Richard Song",
  title: "Autonomous Vision-Guided Precision Landing — Guide",
  description: "Setup, operation, testing and system reference for the eVTOL precision-landing repository",
  numbering: {
    config: [
      {
        reference: "guide-bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 260 } } } }],
      },
      {
        reference: "guide-numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 260 } } } }],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1200, bottom: 1200, left: 1440, right: 1440 },
      },
      titlePage: true,
    },
    headers: {
      first: new Header({ children: [new Paragraph({ children: [] })] }),
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "Vision-Guided Precision Landing — Guide", font: BODY, size: 16, color: MUTED })],
          spacing: { after: 0 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE } },
        })],
      }),
    },
    footers: {
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: BODY, size: 16, color: MUTED })],
        })],
      }),
    },
    children: [...front, ...children],
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(OUT, b);
  console.log("wrote", OUT, (b.length / 1024).toFixed(0) + " KB");
});
