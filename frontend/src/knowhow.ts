import { marked } from "marked";
import { h } from "./dom";
import { langSwitch, t } from "./i18n";

// Obsah je v src/knowhow/<n-skupina>/<nn-sekce>.md (EN, technická dokumentace);
// pořadí skupin i sekcí podle čísla v názvu.
const FILES = import.meta.glob("./knowhow/*/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

type Group = "hardware" | "ingest" | "postproduction" | "project";

interface Section {
  group: Group;
  slug: string;
  title: string;
  md: string;
}

const SECTIONS: Section[] = Object.entries(FILES)
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([path, md]) => ({
    group: /\/\d+-([a-z]+)\/[^/]+$/.exec(path)![1] as Group,
    slug: path.replace(/^.*\/\d+-/, "").replace(/\.md$/, ""),
    title: /^#\s+(.+)$/m.exec(md)?.[1] ?? path,
    md,
  }));

export function renderKnowhow(root: HTMLElement, slug: string | undefined): void {
  const current = SECTIONS.find((s) => s.slug === slug) ?? SECTIONS[0]!;
  const article = h("article", { class: "kh-article" });
  // Vlastní obsah z repozitáře, ne uživatelský vstup → bez sanitizace.
  article.innerHTML = marked.parse(current.md, { gfm: true, async: false });
  // Odkazy ven do nového okna, interní (#/…) normálně.
  for (const a of article.querySelectorAll("a[href^='http']")) {
    a.setAttribute("target", "_blank");
    a.setAttribute("rel", "noopener");
  }

  const d = t();
  const groupLabel: Record<Group, string> = {
    hardware: d.khHardware,
    ingest: d.khIngest,
    postproduction: d.khPostproduction,
    project: d.khProject,
  };
  const groups = [...new Set(SECTIONS.map((s) => s.group))];
  const nav = h(
    "nav",
    { class: "kh-nav", "aria-label": "Know-how" },
    ...groups.flatMap((g) => [
      h("div", { class: "kh-group" }, groupLabel[g]),
      ...SECTIONS.filter((s) => s.group === g).map((s) =>
        h(
          "a",
          { href: `#/knowhow/${s.slug}`, "aria-current": s === current ? "page" : undefined },
          s.title,
        ),
      ),
    ]),
  );

  root.replaceChildren(
    h(
      "div",
      { class: "toolbar" },
      h("a", { href: "#/", class: "back" }, t().backToLibrary),
      h("h1", {}, "Know-how"),
      langSwitch(),
    ),
    h(
      "div",
      { class: "kh" },
      h("div", { class: "kh-side" }, nav, h("p", { class: "hint" }, d.knowhowNote)),
      article,
    ),
  );
  window.scrollTo(0, 0);
}
