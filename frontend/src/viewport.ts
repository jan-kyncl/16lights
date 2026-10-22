/**
 * Zoom a posun náhledu (jen zobrazení, neukládá se do úprav).
 * Měřítko je vztažené k plnému rozlišení snímku: 1 = 1 px exportu na 1 CSS px.
 */
export const MAX_SCALE = 8;

export class Viewport {
  private scale = 1;
  private tx = 0;
  private ty = 0;
  private fitted = true;
  private drag: { id: number; x: number; y: number } | null = null;
  private readonly observer: ResizeObserver;

  constructor(
    private readonly stage: HTMLElement,
    private readonly canvas: HTMLCanvasElement,
    /** Šířka plného rozlišení; canvas může mít menší (preview) rozlišení. */
    private fullWidth: number,
    private readonly onChange: (scale: number) => void,
  ) {
    stage.addEventListener("wheel", this.onWheel, { passive: false });
    stage.addEventListener("pointerdown", this.onDown);
    stage.addEventListener("pointermove", this.onMove);
    stage.addEventListener("pointerup", this.onUp);
    stage.addEventListener("pointercancel", this.onUp);
    stage.addEventListener("dblclick", this.onDblClick);
    this.observer = new ResizeObserver(() => (this.fitted ? this.fit() : this.apply()));
    this.observer.observe(stage);
  }

  destroy(): void {
    this.observer.disconnect();
  }

  private get aspect(): number {
    return this.canvas.height / this.canvas.width;
  }

  /** Nejmenší povolené měřítko (polovina „celého snímku“). */
  get minScale(): number {
    return this.fitScale * 0.5;
  }

  /** Kolik texelů canvasu připadá na jeden fyzický pixel obrazovky. */
  get texelsPerPixel(): number {
    return this.canvas.width / (this.fullWidth * this.scale * devicePixelRatio);
  }

  private get fitScale(): number {
    const w = this.stage.clientWidth - 24;
    const h = this.stage.clientHeight - 24;
    return Math.min(w / this.fullWidth, h / (this.fullWidth * this.aspect));
  }

  /** Po otočení o 90° se mění šířka zobrazení v px plného rozlišení. */
  setFullWidth(w: number): void {
    this.fullWidth = w;
    this.fit();
  }

  fit(): void {
    this.fitted = true;
    this.scale = this.fitScale;
    this.tx = this.ty = 0;
    this.apply();
  }

  /** Zoom na `scale` se zachováním bodu pod (cx, cy) v souřadnicích stage. */
  zoomTo(scale: number, cx = this.stage.clientWidth / 2, cy = this.stage.clientHeight / 2): void {
    const s = Math.min(MAX_SCALE, Math.max(this.minScale, scale));
    const k = s / this.scale;
    const ox = cx - this.stage.clientWidth / 2;
    const oy = cy - this.stage.clientHeight / 2;
    this.tx = ox - (ox - this.tx) * k;
    this.ty = oy - (oy - this.ty) * k;
    this.scale = s;
    this.fitted = false;
    this.apply();
  }

  zoomBy(factor: number): void {
    this.zoomTo(this.scale * factor);
  }

  private apply(): void {
    const w = this.fullWidth * this.scale;
    const h = w * this.aspect;
    // Posun omezit tak, aby obraz neutekl ze stage; menší obraz zůstane na středu.
    const mx = Math.max(0, (w - this.stage.clientWidth) / 2);
    const my = Math.max(0, (h - this.stage.clientHeight) / 2);
    this.tx = Math.min(mx, Math.max(-mx, this.tx));
    this.ty = Math.min(my, Math.max(-my, this.ty));
    const c = this.canvas.style;
    c.width = `${w}px`;
    c.height = `${h}px`;
    c.transform = `translate(calc(-50% + ${this.tx}px), calc(-50% + ${this.ty}px))`;
    // Nad rozlišením náhledu ukázat pixely ostře místo rozmazání.
    c.imageRendering = w > this.canvas.width * 1.5 ? "pixelated" : "auto";
    this.stage.classList.toggle("pannable", mx > 0 || my > 0);
    this.onChange(this.scale);
  }

  private local(ev: MouseEvent): [number, number] {
    const r = this.stage.getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top];
  }

  private onWheel = (ev: WheelEvent): void => {
    ev.preventDefault();
    // Trackpad pinch posílá ctrlKey + jemné deltaY, kolečko myši hrubé kroky.
    const step = ev.ctrlKey ? 0.01 : 0.002;
    this.zoomTo(this.scale * Math.exp(-ev.deltaY * step), ...this.local(ev));
  };

  private onDown = (ev: PointerEvent): void => {
    if (ev.button !== 0 || ev.target !== this.canvas) return;
    this.drag = { id: ev.pointerId, x: ev.clientX, y: ev.clientY };
    this.stage.setPointerCapture(ev.pointerId);
    this.stage.classList.add("dragging");
  };

  private onMove = (ev: PointerEvent): void => {
    if (!this.drag || ev.pointerId !== this.drag.id) return;
    this.tx += ev.clientX - this.drag.x;
    this.ty += ev.clientY - this.drag.y;
    this.drag.x = ev.clientX;
    this.drag.y = ev.clientY;
    this.apply();
  };

  private onUp = (ev: PointerEvent): void => {
    if (!this.drag || ev.pointerId !== this.drag.id) return;
    this.drag = null;
    this.stage.classList.remove("dragging");
  };

  /** Je kurzor nad obrazem? (target bývá kvůli pointer capture stage, ne canvas) */
  private overCanvas(ev: MouseEvent): boolean {
    const r = this.canvas.getBoundingClientRect();
    return (
      ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom
    );
  }

  private onDblClick = (ev: MouseEvent): void => {
    if (!this.overCanvas(ev)) return;
    if (this.fitted) this.zoomTo(1, ...this.local(ev));
    else this.fit();
  };
}
