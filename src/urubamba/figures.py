"""Eight article figures (PNG 300 dpi, SVG with editable text and a caption file each).

Every number that appears in a caption is read from the result tables, so captions
and tables cannot drift apart.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
from scipy.ndimage import distance_transform_edt

from . import config as C
from . import data, features
from .grid import read
from .models import RASTERS, TABLES

FIG = C.FIGURES
INK, BLUE, TEAL, GOLD, RED, GRAY = "#192D3B", "#246B8E", "#168477", "#C58A20", "#B54C60", "#75818A"
EXTENT = [C.BOUNDS[0] / 1e3, C.BOUNDS[2] / 1e3, C.BOUNDS[1] / 1e3, C.BOUNDS[3] / 1e3]
LABELS = {"NDSI": "NDSI", "NDSI_filtro": "NDSI + filtro",
          "ET_base": "Extra Trees base", "ET_base_filtro": "Extra Trees base + filtro",
          "ET_relieve": "Extra Trees + relieve", "ET_relieve_filtro": "Relieve + filtro",
          "ET_prudente": "Extra Trees + semillas prudentes", "ET_prudente_filtro": "Semillas prudentes + filtro",
          "HGB_prudente": "Gradient boosting", "HGB_prudente_filtro": "Gradient boosting + filtro",
          "Ensamble_ET": "Ensamble de árboles", "Ensamble_ET_filtro": "Ensamble + filtro",
          "ET_base_guiado": "Extra Trees base + guiado", "ET_prudente_guiado": "Semillas prudentes + guiado"}
BASELINES = ["NDSI", "NDSI_filtro", "ET_base", "ET_base_filtro"]
STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
         "axes.edgecolor": "#A8B2BA", "axes.linewidth": .6, "axes.spines.top": False,
         "axes.spines.right": False, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
         "ytick.color": INK, "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none",
         "figure.facecolor": "white", "savefig.facecolor": "white"}


def fmt(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def thousands(n):
    return f"{int(n):,}".replace(",", " ")


def date(scene):
    d = scene.split("_")[2]
    return f"{d[6:]}-{d[4:6]}-{d[:4]}"


def export(fig, name, caption):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=.10)
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=.10)
    (FIG / f"{name}.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left", pad=9)


def stretch(cube):
    """One RGB enhancement for every map: clip(reflectance / 0.9, 0, 1) ** 0.55."""
    return np.nan_to_num(np.clip(cube[:, :, [2, 1, 0]] / .9, 0, 1) ** .55)


def mapbase(ax, rgb, window=None):
    ax.imshow(rgb, extent=EXTENT, interpolation="nearest", rasterized=True)
    if window:
        ax.set_xlim(window[:2])
        ax.set_ylim(window[2:])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#D4DADE")
        s.set_linewidth(.5)


def fig_geoprocess(cube, valid):
    fig = plt.figure(figsize=(12, 4.1))
    axes = [fig.add_axes([.025 + i * .245, .40, .215, .46]) for i in range(4)]
    fields = [stretch(cube), np.ma.masked_where(~valid, cube[..., 6]),
              np.ma.masked_where(~valid, read(RASTERS / "pseudoetiquetas_objetivo_ilustracion.tif")),
              np.ma.masked_where(~valid, read(RASTERS / "variable_incidencia.tif"))]
    titles = ["Imagen multiespectral", "Índices espectrales", "Muestras automáticas", "Relieve · entrada auxiliar"]
    notes = ["Sentinel-2 L2A · RGB\nRecorte y alineación a 20 m\nMáscara SCL de píxeles válidos",
             "NDSI = (B03 − B11)/(B03 + B11)\nTambién NDVI, NDWI y NDGI\nContexto local: 3 × 3 y 7 × 7",
             "Positivos y negativos por reglas\nAmbiguos excluidos del ajuste\nLa oscuridad sola no etiqueta fondo",
             "DEM + geometría solar de la escena\nPendiente y orientación de ladera\nSe incorpora a los predictores"]
    for i, (ax, field) in enumerate(zip(axes, fields)):
        if i == 0:
            ax.imshow(field, extent=EXTENT, rasterized=True)
        elif i in (1, 3):
            im = ax.imshow(field, extent=EXTENT, cmap="RdBu" if i == 1 else "cividis", vmin=-1, vmax=1,
                           rasterized=True)
            cax = fig.add_axes([.025 + i * .245, .33, .215, .018])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
            cb.ax.tick_params(labelsize=7, length=2)
        else:
            ax.imshow(field, extent=EXTENT, cmap=ListedColormap([GRAY, TEAL, "#E7ECEF"]), vmin=-.5, vmax=2.5,
                      rasterized=True)
            ax.legend(handles=[Patch(color=TEAL, label="Positivo"), Patch(color=GRAY, label="Negativo"),
                               Patch(color="#E7ECEF", label="Ambiguo")], loc="upper center",
                      bbox_to_anchor=(.5, -.07), ncol=3, frameon=False, fontsize=6.5, handlelength=.8,
                      columnspacing=.7)
        ax.set(xticks=[], yticks=[])
        panel(ax, chr(97 + i), titles[i])
        fig.text(.025 + i * .245, .23, notes[i], fontsize=8, va="top", linespacing=1.65)
    fig.canvas.draw()
    for a, b in zip(axes[:2], axes[1:3]):
        pa, pb = a.get_position(), b.get_position()
        fig.add_artist(FancyArrowPatch((pa.x1 + .005, (pa.y0 + pa.y1) / 2), (pb.x0 - .005, (pb.y0 + pb.y1) / 2),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                       color=INK))
    export(fig, "01_geoprocesamiento",
           f"Figura 1. Construcción de variables y pseudoetiquetas. (a–c) Reflectancias, índices y reglas de "
           f"etiquetado; (d) DEM y geometría solar como entradas auxiliares. Las cuatro vistas muestran la escena "
           f"objetivo del {date(C.TARGET_SCENE)} para explicar el procesamiento; sus pseudoetiquetas NO se usan para "
           f"entrenar. La misma transformación se aplica a las cuatro fechas de ajuste y reúne "
           f"{len(features.FULL)} variables. La incidencia describe la iluminación local de la ladera, no sombras "
           f"proyectadas ni una corrección radiométrica. El DEM promediado a 20 m no gana resolución efectiva. "
           f"Norte arriba; recorte de 22 × 14,5 km.")


def fig_design(n_support):
    plt.rcParams.update({"font.size": 10})
    ink, line, blue, green, ochre = "#23343E", "#53636B", "#315F7B", "#45736A", "#AE853C"
    fig = plt.figure(figsize=(12, 9.3))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")

    def txt(x, y, s, size=10, weight="normal", ha="left", color=ink, **kw):
        fig.text(x, y, s, fontsize=size, weight=weight, ha=ha, va="center", color=color, **kw)

    def box(x, y, w, h, fill="#F5F7F8", dash=False):
        fig.add_artist(Rectangle((x, y), w, h, facecolor=fill, edgecolor=line, lw=1,
                                 linestyle=(0, (4, 3)) if dash else "-", transform=fig.transFigure))

    def arrow(points, color=line, dash=False, head=True):
        segs = points[:-1] if head else points
        for a, b in zip(segs[:-1], segs[1:]):
            fig.add_artist(plt.Line2D([a[0], b[0]], [a[1], b[1]], transform=fig.transFigure, color=color,
                                      lw=1.1, ls="--" if dash else "-"))
        if head:
            fig.add_artist(FancyArrowPatch(points[-2], points[-1], transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=10, lw=1.1, color=color, linestyle="--" if dash else "-"))

    def thumb(rect, arr, circle=False, cmap=None, vmin=None, vmax=None):
        ax = fig.add_axes(rect, zorder=3)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
        ax.axis("off")
        if circle:
            clip = Circle((.5, .5), .486, transform=ax.transAxes, facecolor="none", edgecolor=ink, lw=.9)
            ax.add_patch(clip)
            im.set_clip_path(clip)
        return ax

    fit_scene = C.FIT_SCENES[-1]
    dem = data.load_dem()
    cube, valid = data.load_scene(fit_scene)
    valid &= np.isfinite(dem)
    feats = features.full(cube, valid, dem, data.sun_geometry(fit_scene))
    cut = np.s_[245:545, 725:1025]
    rgb = stretch(cube)
    txt(.045, .972, "De la imagen al mapa de nieve y hielo", 15, weight="bold")
    txt(.045, .939, "URUBAMBA  /  DISEÑO DEL EXPERIMENTO", 8.5, color=line)
    txt(.955, .939, "Sentinel-2 · 20 m", 9, ha="right", color=line)
    fig.add_artist(plt.Line2D([.045, .955], [.920, .920], transform=fig.transFigure, color="#B6C1C6", lw=.8))
    box(.045, .812, .30, .085, "#EAF1F6", True)
    txt(.07, .866, "Sentinel-2 L2A", 11, weight="bold")
    txt(.07, .839, f"Cuatro fechas de ajuste · {C.FIT_SCENES[0][10:14]}", 8.5)
    box(.385, .812, .30, .085, "#EDF3ED", True)
    txt(.41, .866, "DEM + geometría solar", 10.5, weight="bold")
    txt(.41, .839, "Relieve e iluminación local", 8.5)
    arrow([(.195, .812), (.195, .787), (.365, .787), (.365, .767)])
    arrow([(.535, .812), (.535, .787), (.365, .787)], head=False)
    for x, arr, cmap, lo, hi in [(.095, rgb[cut], None, None, None), (.290, cube[cut][..., 6], "RdBu", -1, 1),
                                 (.485, feats[cut][..., 15], "cividis", -1, 1)]:
        thumb([x, .615, .15, .15], arr, True, cmap, lo, hi)
    for x, title in [(.170, "Reflectancia RGB"), (.365, "Índices espectrales"), (.560, "Incidencia solar")]:
        txt(x, .603, title, 9, ha="center")
    txt(.365, .581, f"Misma ventana · imagen de ajuste del {date(fit_scene)}", 8, ha="center", color=line)
    box(.045, .522, .640, .039, "#F5F6F6")
    txt(.365, .542, f"{len(features.FULL)} predictores  +  pseudoetiquetas espectrales", 10, ha="center",
        weight="bold")
    arrow([(.365, .575), (.365, .563)])
    arrow([(.365, .522), (.365, .500)])
    box(.045, .352, .640, .147, "#F7F8F8")
    b = C.INNER_VALIDATION["block_px"]
    rr, cc = np.indices(valid.shape)
    val = ((cc // b + 2 * (rr // b)) % C.INNER_VALIDATION["modulo"]) == 0
    train = distance_transform_edt(~val, sampling=C.RESOLUTION) > C.INNER_VALIDATION["gap_m"]
    x = C.BOUNDS[0] + (np.arange(valid.shape[1]) + .5) * C.RESOLUTION
    lo, hi = C.SECTOR_EDGES[1], C.SECTOR_EDGES[2]
    domain = np.broadcast_to((x < lo - C.SPATIAL_GAP_M) | (x >= hi + C.SPATIAL_GAP_M), valid.shape)
    part = np.zeros(valid.shape, int)
    part[:, (x >= lo) & (x < hi)] = 3
    part[domain & train] = 1
    part[domain & val] = 2
    ax = thumb([.058, .365, .195, .122], rgb)
    ax.imshow(part, cmap=ListedColormap(["#D6DEE1", blue, ochre, green]), vmin=-.5, vmax=3.5, alpha=.70,
              interpolation="nearest", rasterized=True)
    txt(.274, .476, "Validación espacial anidada", 11, weight="bold")
    txt(.274, .447, f"Sector externo excluido + margen de {C.SPATIAL_GAP_M} m", 8.7)
    txt(.274, .423, f"Bloques internos de {b * C.RESOLUTION // 1000} km + margen de "
                    f"{C.INNER_VALIDATION['gap_m']} m", 8.7)
    txt(.274, .397, "Rotación: oeste / centro / este", 8.7)
    for xx, col, label in [(.274, blue, "Ajuste"), (.385, ochre, "Validación"), (.531, green, "Sector excluido")]:
        fig.add_artist(Rectangle((xx, .368), .009, .009, facecolor=col, edgecolor="none", transform=fig.transFigure))
        txt(xx + .014, .373, label, 7.5)
    arrow([(.365, .352), (.365, .332), (.198, .332), (.198, .316)])
    arrow([(.365, .332), (.532, .332), (.532, .316)])
    box(.045, .241, .306, .075, "#EAF1F6")
    txt(.198, .290, "Extra Trees", 11, weight="bold", ha="center")
    txt(.198, .263, f"{C.EXTRA_TREES['n_estimators']} árboles · hoja mínima {C.EXTRA_TREES['min_samples_leaf']}",
        8.7, ha="center")
    box(.379, .241, .306, .075, "#EAF1F6")
    txt(.532, .290, "Gradient boosting", 11, weight="bold", ha="center")
    txt(.532, .263, "Iteración elegida por log-loss interno", 8.7, ha="center")
    arrow([(.198, .241), (.198, .222), (.365, .222), (.365, .202)])
    arrow([(.532, .241), (.532, .222), (.365, .222)], head=False)
    box(.045, .130, .640, .071, "#EDF3ED")
    txt(.365, .176, "Predicción por sectores y composición del mapa", 10.5, ha="center", weight="bold")
    txt(.365, .149, "Reajuste por pliegue · umbral 0,50 · ensamble y filtros", 8.5, ha="center")
    arrow([(.365, .130), (.365, .101)])
    box(.045, .025, .640, .075, "#F5EEEE")
    txt(.365, .074, "Evaluación de las predicciones congeladas", 10.5, ha="center", weight="bold")
    txt(.365, .046, f"IoU · F1 · AP · borde · área    /    {thousands(n_support)} píxeles evaluados", 8.7,
        ha="center")
    box(.752, .635, .203, .262, "#F6F8F9", True)
    txt(.8535, .875, "Imagen objetivo", 10.5, weight="bold", ha="center")
    txt(.8535, .850, date(C.TARGET_SCENE), 8.5, ha="center")
    target, _ = data.load_scene(C.TARGET_SCENE)
    thumb([.791, .674, .125, .125 * 12 / 9.3], stretch(target)[cut], True)
    txt(.8535, .652, "Entrada de inferencia", 8.5, ha="center", color=blue)
    arrow([(.752, .685), (.720, .685), (.720, .165), (.687, .165)], blue)
    box(.778, .356, .177, .164, "#F5EEEE", True)
    txt(.8665, .496, "Referencia externa", 10, weight="bold", ha="center")
    txt(.8665, .468, "Contornos Landsat 2020", 8.2, ha="center")
    txt(.8665, .426, "Se usa únicamente\npara contrastar el mapa", 8.4, ha="center", linespacing=1.6)
    txt(.8665, .378, "Evaluación retrospectiva", 7.7, ha="center", color="#82575B")
    arrow([(.8665, .356), (.8665, .063), (.687, .063)], "#82575B", True)
    export(fig, "02_diseno_entrenamiento",
           f"Figura 2. Flujo del experimento. Los recortes circulares proceden de la adquisición de ajuste del "
           f"{date(fit_scene)}. La imagen objetivo del {date(C.TARGET_SCENE)} entra sólo para inferencia. Se generan "
           f"pseudoetiquetas espectrales en cuatro fechas de ajuste y se extraen {len(features.FULL)} predictores. "
           f"La partición ilustrada excluye el sector central y {C.SPATIAL_GAP_M} m de margen; los bloques internos "
           f"de validación miden {b * C.RESOLUTION // 1000} km, con separación de {C.INNER_VALIDATION['gap_m']} m. "
           f"Se rota el sector externo para formar el mapa completo. El mínimo log-loss interno elige la iteración "
           f"del boosting; después se reajusta con todas las muestras permitidas. Los contornos Landsat sólo "
           f"intervienen en la evaluación. Los colores del mapa de partición indican asignaciones geográficas, no "
           f"clases. Norte arriba; paneles a distintas escalas.")
    plt.rcParams.update(STYLE)


def fig_learning():
    curves = pd.read_csv(TABLES / "entrenamiento_boosting.csv")
    sel = pd.read_csv(TABLES / "seleccion_iteraciones.csv").set_index("sector")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.985, top=.85, bottom=.23, wspace=.15)
    ymin = max(1e-6, 10 ** np.floor(np.log10(curves[["train_loss", "val_loss"]].min().min())))
    for fold, ax in enumerate(axes):
        d = curves[curves.sector == fold]
        s = sel.loc[fold]
        k = int(s.selected_iteration)
        ax.plot(d.iteration, d.train_loss, color=BLUE, lw=1.5, label="Ajuste")
        ax.plot(d.iteration, d.val_loss, color=GOLD, lw=1.5, label="Validación espacial")
        ax.axvline(k, color=GRAY, lw=.8, ls="--")
        ax.scatter([k], [s.validation_logloss], s=28, color=GOLD, zorder=3, edgecolors="white", linewidths=.5)
        ax.text(.97, .95, f"Iteración elegida: {k}\nn ajuste = {thousands(s.inner_train)}\n"
                          f"n validación = {thousands(s.inner_val)}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, linespacing=1.6, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
        panel(ax, chr(97 + fold), f"Sector externo: {C.SECTOR_NAMES[fold].lower()}")
        ax.set(xlabel="Iteración de boosting", xlim=(1, C.BOOSTING["max_iter"]), yscale="log", ylim=(ymin, 1))
        ax.grid(axis="y", color="#E7EBEE", lw=.6)
    axes[0].set_ylabel("Log-loss · escala logarítmica ↓")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .015), ncol=2,
               frameon=False)
    its = ", ".join(str(int(v)) for v in sel.selected_iteration)
    export(fig, "03_curvas_entrenamiento",
           f"Figura 3. Pérdida por iteración del HistGradientBoostingClassifier en escala logarítmica. Azul: "
           f"ajuste; ocre: validación en bloques geográficos excluidos. El mínimo de validación elige la iteración "
           f"({its} para oeste, centro y este) antes del reajuste en toda el área permitida. Las tres curvas son "
           f"tres modelos distintos, no repeticiones. Miden fidelidad a las pseudoetiquetas, no exactitud glaciar: "
           f"los píxeles ambiguos no entran en esta pérdida.")


def fig_comparison(tab, best, n_support):
    names = list(dict.fromkeys(["NDSI", "NDSI_filtro", "ET_base_filtro", "ET_relieve_filtro",
                                "ET_prudente_filtro", "HGB_prudente_filtro", best]))
    tt = tab.loc[names]
    y = np.arange(len(tt))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), sharey=True, gridspec_kw={"width_ratios": [1.35, 1, 1]})
    fig.subplots_adjust(left=.29, right=.98, top=.87, bottom=.20, wspace=.30)
    for i, (name, row) in enumerate(tt.iterrows()):
        color = TEAL if name == best else BLUE if name == "ET_base_filtro" else GRAY
        axes[0].errorbar(row.iou, y[i], xerr=[[row.iou - row.iou_lo], [row.iou_hi - row.iou]], fmt="o",
                         color=color, ms=5, capsize=2, lw=1.)
        axes[0].annotate(fmt(row.iou), (row.iou, y[i]), xytext=(6, -11), textcoords="offset points", fontsize=7,
                         color=color)
        axes[1].scatter(row.f1_borde_20m, y[i], color=color, s=25)
        axes[2].scatter(row.sesgo_area_km2, y[i], color=color, s=25)
    for ax in axes:
        ax.grid(axis="x", color="#E7EBEE", lw=.6)
        ax.set_ylim(len(y) - .45, -.6)
    axes[0].set(yticks=y, yticklabels=[LABELS[n] for n in names], xlabel="IoU ↑",
                xlim=(tt.iou_lo.min() - .02, tt.iou_hi.max() + .02))
    axes[1].set(xlabel="F1 de borde · 20 m ↑", xlim=(tt.f1_borde_20m.min() - .025, tt.f1_borde_20m.max() + .025))
    axes[2].axvline(0, color=INK, lw=.7, ls="--")
    axes[2].set(xlabel="Sesgo de área (km²)",
                xlim=(min(0, tt.sesgo_area_km2.min()) - .1, max(0, tt.sesgo_area_km2.max()) + .12))
    for ax, l, t in zip(axes, "abc", ["Solapamiento", "Localización del borde", "Área predicha − referencia"]):
        panel(ax, l, t)
    fig.text(.29, .055, f"Barras: intervalo descriptivo del 95 % · {thousands(C.BOOTSTRAP['draws'])} remuestras "
                        "pareadas de bloques de 1 km.\nOrden metodológico fijo; la IoU más alta se destaca en verde.",
             fontsize=8, color=GRAY)
    export(fig, "04_comparacion_controlada",
           f"Figura 4. Comparación sobre los mismos {thousands(n_support)} píxeles de 20 m. Se muestra una "
           f"selección por familia; la tabla completa incluye todos los candidatos, también los que empeoran. IoU y "
           f"F1 de borde son métricas distintas: más área coincidente no implica mejor límite. Ejes ampliados para "
           f"distinguir diferencias pequeñas. El candidato destacado se elige después de ver la evaluación y sus "
           f"intervalos no corrigen esa selección.")


def fig_ablations():
    abl = pd.read_csv(TABLES / "ablaciones.csv")
    imp = pd.read_csv(TABLES / "importancia_grupos.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.24, right=.97, bottom=.24, top=.86, wspace=.85)
    for i, row in abl.iterrows():
        ax.errorbar(row.delta_iou * 100, i, xerr=[[(row.delta_iou - row.lo) * 100], [(row.hi - row.delta_iou) * 100]],
                    fmt="o", color=TEAL if row.delta_iou >= 0 else RED, ms=4, capsize=2)
    ax.axvline(0, color=GRAY, lw=.8, ls="--")
    ax.set(yticks=range(len(abl)), yticklabels=abl.pregunta, ylim=(len(abl) - .5, -.5),
           xlabel="Diferencia de IoU (puntos)")
    ax.grid(axis="x", color="#E7EBEE", lw=.6)
    groups = list(features.GROUPS)
    for i, g in enumerate(groups):
        vals = imp[imp.grupo == g].groupby("sector").delta_log_loss.mean()
        bx.scatter(vals, np.repeat(i, len(vals)), c=[BLUE, GOLD, TEAL][:len(vals)], s=23, zorder=3)
        bx.plot([vals.min(), vals.max()], [i, i], color="#BBC3C9", lw=1, zorder=1)
    bx.axvline(0, color=GRAY, lw=.8, ls="--")
    bx.set(yticks=range(3), yticklabels=[g.replace(" e ", "\ne ").replace(" espacial", "\nespacial") for g in groups],
           ylim=(2.6, -.6), xlabel="Aumento de log-loss al permutar")
    bx.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=s)
                       for c, s in zip([BLUE, GOLD, TEAL], C.SECTOR_NAMES)], loc="lower right", frameon=False,
              fontsize=7)
    panel(ax, "a", "Aporte de cada decisión")
    panel(bx, "b", "Dependencia del boosting")
    fig.text(.24, .04, "(a) Intervalos pareados por bloques sobre la referencia externa.\n(b) Promedio de 3 "
                       "permutaciones por grupo y sector; validación interna de pseudoetiquetas.", fontsize=8,
             color=GRAY)
    export(fig, "05_ablaciones_e_importancia",
           "Figura 5. Ablaciones e importancia por grupos. La primera ablación añade a la vez relieve, NDGI y "
           "contexto 7 × 7, así que no separa el efecto de cada variable. Las semillas prudentes eliminan la regla "
           "que convertía en negativo todo píxel oscuro. El filtro guiado conserva discontinuidades de NDSI y usa el "
           "mismo mínimo de componente (1 ha), sin mayoría 3 × 3. Importancia: cambio de log-loss al permutar "
           "conjuntamente cada grupo, sólo en la validación interna; no es causalidad ni aporte a la exactitud "
           "externa.")


def fig_errors(tab, best, rgb, truth, support):
    base = read(RASTERS / "mascara_ET_base_filtro.tif") == 1
    new = read(RASTERS / f"mascara_{best}.tif") == 1
    windows = [(808, 815, 8537, 8544), (815, 822, 8531, 8538), (822.5, 829.5, 8532.5, 8539.5)]
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 10.2))
    fig.subplots_adjust(left=.06, right=.985, bottom=.09, top=.945, wspace=.035, hspace=.18)
    xs = np.linspace(EXTENT[0] + .01, EXTENT[1] - .01, truth.shape[1])
    ys = np.linspace(EXTENT[3] - .01, EXTENT[2] + .01, truth.shape[0])
    for r, win in enumerate(windows):
        for c, ax in enumerate(axes[r]):
            mapbase(ax, rgb, win)
            if c == 0:
                ax.contour(xs, ys, truth & support, levels=[.5], colors=["#59CBD8"], linewidths=.65)
            else:
                pred = base if c == 1 else new
                err = np.where(~truth & pred, 1, np.where(truth & ~pred, 2, 0))
                ax.imshow(np.ma.masked_where((err == 0) | ~support, err), extent=EXTENT,
                          cmap=ListedColormap([GOLD, RED]), vmin=.5, vmax=2.5, alpha=.9, interpolation="nearest",
                          rasterized=True)
            ax.imshow(np.ma.masked_where(support, np.ones_like(support)), extent=EXTENT,
                      cmap=ListedColormap(["#D6DBDF"]), alpha=.6, vmin=0, vmax=1, rasterized=True)
            panel(ax, chr(97 + r * 3 + c), ["RGB + referencia", "Errores · base", "Errores · destacado"][c])
            if c == 0:
                ax.set_ylabel(C.SECTOR_NAMES[r].upper(), fontsize=10, labelpad=8)
                x0, y0 = win[0] + .35, win[2] + .4
                ax.plot([x0, x0 + 1], [y0, y0], color="white", lw=2)
                ax.text(x0 + .5, y0 + .12, "1 km", ha="center", color="white", fontsize=7)
            if r == 0 and c == 0:
                ax.annotate("N", xy=(.92, .92), xytext=(.92, .75), xycoords="axes fraction", color="white",
                            ha="center", fontsize=9, arrowprops=dict(arrowstyle="-|>", color="white", lw=1))
    fig.legend(handles=[Line2D([], [], color="#59CBD8", label="Referencia"), Patch(color=GOLD, label="Comisión"),
                        Patch(color=RED, label="Omisión"), Patch(color="#D6DBDF", label="Sin evaluación")],
               loc="lower center", bbox_to_anchor=(.5, .042), ncol=4, frameon=False, fontsize=8)
    fig.text(.06, .015, f"Base: {LABELS['ET_base_filtro']}. Destacado: {LABELS[best]}. EPSG:32718 · píxel 20 m · "
                        "realce RGB común.", fontsize=8, color=GRAY)
    export(fig, "06_evidencia_espacial",
           "Figura 6. Ventanas de 7 × 7 km, una por sector, fijadas antes de evaluar. Límites en km UTM 18S: oeste "
           "E 808–815 / N 8537–8544; centro E 815–822 / N 8531–8538; este E 822,5–829,5 / N 8532,5–8539,5. Primera "
           "columna: imagen y contorno Landsat de referencia; segunda y tercera: sólo desacuerdos, con la imagen "
           "visible donde coinciden. Naranja: comisión; magenta: omisión. Realce RGB común = clip(reflectancia/0,9, "
           "0, 1)^0,55. El compuesto Landsat tiene otro soporte temporal y otra resolución efectiva. Norte de "
           "grilla arriba.")


def fig_regimes(best):
    reg = pd.read_csv(TABLES / "regimenes.csv")
    chosen = list(dict.fromkeys(["NDSI", "ET_base_filtro", best]))
    cats = ["Incidencia < 0,3", "Incidencia ≥ 0,3", "Pendiente > 30°", "Pendiente ≤ 30°", "NIR < 0,15", "NIR ≥ 0,15"]
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.7), sharey=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.26, top=.86, wspace=.14)
    for i, ax in enumerate(axes):
        pair = cats[2 * i:2 * i + 2]
        for j, (name, color, marker) in enumerate(zip(chosen, [GRAY, BLUE, TEAL], ["o", "s", "D"])):
            vals = reg[reg.metodo == name].set_index("regimen").loc[pair]
            ax.scatter(np.array([0, 1]) + (j - 1) * .16, vals.iou, color=color, marker=marker, s=28,
                       label=LABELS[name])
        counts = reg[reg.metodo == "NDSI"].set_index("regimen").loc[pair, "n_positivos"]
        ax.set(xticks=[0, 1], xticklabels=[f"{s}\n{thousands(n)} positivos" for s, n in zip(pair, counts)],
               xlim=(-.5, 1.5), ylim=(0, 1))
        ax.grid(axis="y", color="#E7EBEE", lw=.6)
        panel(ax, chr(97 + i), ["Iluminación local", "Pendiente del terreno", "Respuesta infrarroja"][i])
    axes[0].set_ylabel("IoU respecto de la referencia ↑")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .005), frameon=False,
               ncol=3, fontsize=8)
    export(fig, "07_condiciones_dificiles",
           "Figura 7. Rendimiento según terreno y señal de la imagen. Cada par divide los mismos píxeles evaluados; "
           "las tres familias se superponen y no se suman entre sí. Se indica el número de positivos de referencia "
           "para leer con cuidado los grupos pequeños. Incidencia < 0,3 significa iluminación local baja, no una "
           "máscara completa de sombras. Los umbrales no se reajustaron con estos resultados.")


def fig_tradeoffs(tab):
    curves = pd.read_csv(TABLES / "precision_recall.csv")
    aps = pd.read_csv(TABLES / "average_precision.csv").set_index("metodo")
    top_ap = aps.drop(index="NDSI").AP.idxmax()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    fig.subplots_adjust(left=.07, right=.98, top=.87, bottom=.20, wspace=.34)
    shown = list(dict.fromkeys(["NDSI", "ET_base", "Ensamble_ET", top_ap]))
    for name, color, ls in zip(shown, [GRAY, GRAY, TEAL, BLUE], [":", "--", "-.", "-"]):
        c = curves[curves.metodo == name]
        ax.step(c.recall, c.precision, where="post", color=color, ls=ls, lw=1.4,
                label=f"{LABELS[name]} · AP {fmt(aps.loc[name, 'AP'])}")
    prevalence = aps.prevalencia.iloc[0]
    ax.axhline(prevalence, lw=.7, color=GRAY, ls=":")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precisión")
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, .08))
    points = [("NDSI_filtro", GRAY), ("ET_base_filtro", GRAY), ("Ensamble_ET_filtro", TEAL),
              ("ET_base_guiado", GOLD), ("HGB_prudente_filtro", BLUE)]
    sub = tab.loc[[p for p, _ in points]]
    for (name, color), offset in zip(points, [(8, -14), (-10, 11), (8, 7), (8, 7), (8, -15)]):
        row = tab.loc[name]
        bx.scatter(row.iou, row.f1_borde_20m, s=32, color=color)
        bx.annotate(LABELS[name], (row.iou, row.f1_borde_20m), xytext=offset, textcoords="offset points", fontsize=7,
                    ha="right" if offset[0] < 0 else "left")
    pad_x = max(.004, (sub.iou.max() - sub.iou.min()) * .35)
    pad_y = max(.008, (sub.f1_borde_20m.max() - sub.f1_borde_20m.min()) * .35)
    bx.set(xlim=(sub.iou.min() - pad_x, sub.iou.max() + pad_x),
           ylim=(sub.f1_borde_20m.min() - pad_y, sub.f1_borde_20m.max() + pad_y), xlabel="IoU ↑",
           ylabel="F1 de borde · 20 m ↑")
    for a in (ax, bx):
        a.grid(color="#E7EBEE", lw=.6)
    panel(ax, "a", "Calidad del puntaje continuo")
    panel(bx, "b", "Solapamiento y borde: objetivos distintos")
    fig.text(.07, .045, "AP se calcula antes del filtro espacial; los puntajes no son probabilidades calibradas.\n"
                        "El umbral 0,50 se mantuvo fijo: no se buscó el que maximiza esta evaluación.", fontsize=8,
             color=GRAY)
    export(fig, "08_puntajes_y_compromisos",
           f"Figura 8. (a) Curvas precisión–recall completas y escalonadas; su integral coincide con la average "
           f"precision (AP). Prevalencia positiva {fmt(prevalence * 100, 2)} %; la línea punteada horizontal es "
           f"el AP de un puntaje aleatorio. (b) Solapamiento frente a límite a 20 m, con ejes ampliados. Una AP "
           f"mayor indica que el puntaje ordena mejor los píxeles frente a esta referencia; no garantiza una mejora "
           f"equivalente con el umbral fijo.")


def main():
    plt.rcParams.update(STYLE)
    tab = pd.read_csv(TABLES / "metricas.csv").set_index("metodo")
    candidates = [n for n in tab.index if n not in BASELINES]
    best = tab.loc[candidates].iou.idxmax()
    (TABLES / "candidato_destacado.txt").write_text(best, encoding="utf-8")
    cube, valid = data.load_scene(C.TARGET_SCENE)
    support = read(RASTERS / "valid_target.tif").astype(bool)
    truth = read(RASTERS / "referencia_2020.tif").astype(bool)
    n_support = int(support.sum())
    fig_geoprocess(cube, valid)
    fig_design(n_support)
    fig_learning()
    fig_comparison(tab, best, n_support)
    fig_ablations()
    fig_errors(tab, best, stretch(cube), truth, support)
    fig_regimes(best)
    fig_tradeoffs(tab)
    print("Figuras exportadas; candidato destacado:", best, flush=True)
