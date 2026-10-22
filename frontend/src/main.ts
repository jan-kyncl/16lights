import "./style.css";
import { renderEditor } from "./editor";
import { h } from "./dom";
import { t } from "./i18n";
import { renderIngest } from "./ingestPage";
import { renderKnowhow } from "./knowhow";
import { renderLibrary } from "./library";
import { renderModules } from "./modulesPage";

const root = document.getElementById("app") as HTMLElement;

async function route(): Promise<void> {
  const m = /^#\/shot\/([0-9a-f]{64})$/.exec(location.hash);
  root.ondragover = root.ondragleave = root.ondrop = null;
  try {
    const kh = /^#\/knowhow(?:\/([a-z0-9-]+))?$/.exec(location.hash);
    const mods = /^#\/shot\/([0-9a-f]{64})\/modules$/.exec(location.hash);
    if (mods?.[1]) await renderModules(root, mods[1]);
    else if (m?.[1]) await renderEditor(root, m[1]);
    else if (kh) renderKnowhow(root, kh[1]);
    else if (location.hash === "#/ingest") await renderIngest(root);
    else await renderLibrary(root);
  } catch (e) {
    root.replaceChildren(
      h(
        "div",
        { class: "fatal" },
        h("p", {}, (e as Error).message),
        h("a", { href: "#/" }, t().backToLibrary),
      ),
    );
  }
}

window.addEventListener("hashchange", () => void route());
void route();
