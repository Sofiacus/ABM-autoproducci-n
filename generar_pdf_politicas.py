"""
generar_pdf_politicas.py
Genera un PDF descriptivo de las seis políticas públicas simuladas en el ABM
de autoproducción de vivienda (Quintil 1, ENIGH 2024).
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus.flowables import HRFlowable

# ── Paleta ────────────────────────────────────────────────────────────────────
AZUL    = colors.HexColor("#0BA1DA")
VERDE   = colors.HexColor("#BFD602")
NARANJA = colors.HexColor("#F5A623")
ROJO    = colors.HexColor("#D0021B")
NEGRO   = colors.HexColor("#1A1A2E")
GRIS    = colors.HexColor("#6B6B6B")
GRIS_CLR= colors.HexColor("#F7F9FC")
ROSA    = colors.HexColor("#E91E8C")
MORADO  = colors.HexColor("#9C27B0")
NARANJA2= colors.HexColor("#FF6F00")
VERDE2  = colors.HexColor("#00897B")
AZUL2   = colors.HexColor("#1565C0")
VERDE3  = colors.HexColor("#2E7D32")

POL_COLS = {
    "A": ROSA,
    "B": MORADO,
    "C": NARANJA2,
    "D": VERDE2,
    "E": AZUL2,
    "F": VERDE3,
}

# ── Estilos ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

titulo_doc = ParagraphStyle(
    "TituloDoc", parent=styles["Title"],
    fontSize=22, leading=28, textColor=NEGRO,
    spaceAfter=6, alignment=TA_CENTER,
)
subtitulo_doc = ParagraphStyle(
    "SubtituloDoc", parent=styles["Normal"],
    fontSize=11, leading=16, textColor=GRIS,
    spaceAfter=4, alignment=TA_CENTER,
)
seccion = ParagraphStyle(
    "Seccion", parent=styles["Heading1"],
    fontSize=13, leading=18, textColor=colors.white,
    spaceBefore=0, spaceAfter=0,
    leftIndent=10,
)
subseccion = ParagraphStyle(
    "Subseccion", parent=styles["Heading2"],
    fontSize=10, leading=14, textColor=NEGRO,
    spaceBefore=8, spaceAfter=4,
    fontName="Helvetica-Bold",
)
cuerpo = ParagraphStyle(
    "Cuerpo", parent=styles["Normal"],
    fontSize=9.5, leading=14, textColor=NEGRO,
    spaceAfter=5, alignment=TA_JUSTIFY,
)
bullet_style = ParagraphStyle(
    "Bullet", parent=styles["Normal"],
    fontSize=9.5, leading=13, textColor=NEGRO,
    leftIndent=14, firstLineIndent=-10,
    spaceAfter=3,
)
caption = ParagraphStyle(
    "Caption", parent=styles["Normal"],
    fontSize=8.5, leading=12, textColor=GRIS,
    spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Oblique",
)
param_label = ParagraphStyle(
    "ParamLabel", parent=styles["Normal"],
    fontSize=8.5, leading=11, textColor=NEGRO,
    fontName="Helvetica-Bold",
)
param_val = ParagraphStyle(
    "ParamVal", parent=styles["Normal"],
    fontSize=8.5, leading=11, textColor=GRIS,
    fontName="Helvetica",
)
nota = ParagraphStyle(
    "Nota", parent=styles["Normal"],
    fontSize=8, leading=11, textColor=GRIS,
    leftIndent=6, spaceAfter=4,
    fontName="Helvetica-Oblique",
)

W, H = A4  # 595 × 842 pts


# ── Utilidades ────────────────────────────────────────────────────────────────
def b(txt): return f"<b>{txt}</b>"
def i(txt): return f"<i>{txt}</i>"
def bullet(txt): return Paragraph(f"• {txt}", bullet_style)

def header_pol(codigo, nombre, color):
    """Encabezado coloreado de cada política."""
    tbl = Table(
        [[Paragraph(f"<b>{codigo} · {nombre}</b>", seccion)]],
        colWidths=[W - 4*cm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    return tbl

def param_table(rows, color):
    """Tabla compacta de parámetros del modelo."""
    data = [[Paragraph(b("Parámetro"), param_label),
             Paragraph(b("Efecto a intensidad media (50%)"), param_label),
             Paragraph(b("Efecto máximo (100%)"), param_label)]]
    for r in rows:
        data.append([
            Paragraph(r[0], param_val),
            Paragraph(r[1], param_val),
            Paragraph(r[2], param_val),
        ])
    tbl = Table(data, colWidths=[5.5*cm, 6.5*cm, 5.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), color),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, GRIS_CLR]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("ALIGN",        (0,0), (-1,-1), "LEFT"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]))
    return tbl

def hr(color=GRIS):
    return HRFlowable(width="100%", thickness=0.5,
                      color=color, spaceAfter=6, spaceBefore=2)


# ── Contenido de las políticas ────────────────────────────────────────────────
POLITICAS = [
    # ─────────────────────────────────────────────────────────────────────────
    dict(
        codigo="A",
        nombre="Subsidio directo a construcción",
        color=POL_COLS["A"],
        objetivo=(
            "Reducir la barrera financiera que impide a los hogares del quintil 1 "
            "avanzar en la construcción progresiva de su vivienda, mediante "
            "transferencias directas vinculadas a metas constructivas verificables."
        ),
        problema=(
            "Sin excedente de ingreso no hay ahorro posible, y sin capital semilla "
            "la autoproducción nunca arranca. Los costos de construcción —especialmente "
            "en la Etapa 3 (terminaciones)— superan la capacidad de ahorro anual de "
            "la mayoría de hogares del quintil 1, generando un cuello de botella "
            "estructural."
        ),
        acciones=[
            f"{b('Transferencias condicionadas a construcción:')} Pagos directos al hogar al comprobar avance en la obra mediante inspección técnica o fotografía georreferenciada.",
            f"{b('Vales de materiales:')} Subsidio en especie canjeable en ferreterías y distribuidores de materiales de construcción certificados, evitando desvío de recursos.",
            f"{b('Fondo de garantía para vivienda progresiva:')} Respaldo gubernamental que permite al hogar acceder a crédito puente mientras acumula el ahorro necesario.",
            f"{b('Subsidio diferenciado por etapa:')} Mayor apoyo en Etapa 3 (terminaciones) donde los costos son máximos, y apoyo estándar en etapas previas.",
            f"{b('Apoyo técnico gratuito:')} Asesoría de arquitecto o técnico de obra vinculada al subsidio, garantizando calidad y eficiencia en el uso del recurso.",
        ],
        actores=[
            "CONAVI / Secretaría de Desarrollo Agrario, Territorial y Urbano (SEDATU)",
            "Gobiernos estatales y municipales (verificación y padrones)",
            "Institutos estatales de vivienda (OREVIS)",
        ],
        mecanismo=(
            "En el ABM, el subsidio se modela como una reducción directa del costo "
            "de construcción por etapa: hasta 40% en Etapa 3 y 28% en las demás "
            "etapas, a intensidad máxima. Esto incrementa la probabilidad de transición "
            "al reducir la brecha entre ahorro acumulado y costo requerido."
        ),
        params=[
            ("Costo E3", "−20% del costo base", "−40% del costo base"),
            ("Costo E1, E2, E4", "−14% del costo base", "−28% del costo base"),
        ],
        impacto=(
            "Impacto moderado (Δ ≈ −1.3 años a 50% intensidad). Efectivo para "
            "desbloquear el cuello de botella de terminaciones, pero limitado si "
            "el hogar no tiene ingreso suficiente para el ahorro previo."
        ),
    ),
    # ─────────────────────────────────────────────────────────────────────────
    dict(
        codigo="B",
        nombre="Microcrédito para vivienda",
        color=POL_COLS["B"],
        objetivo=(
            "Ampliar la capacidad de ahorro e inversión en construcción mediante "
            "esquemas de crédito accesibles, con condiciones adaptadas al perfil de "
            "ingresos irregulares del quintil 1."
        ),
        problema=(
            "Las instituciones financieras formales excluyen a los hogares de menores "
            "ingresos por falta de historial crediticio, garantías o ingresos formales "
            "comprobables. La informalidad laboral hace que el 80%+ del quintil 1 no "
            "pueda acceder a crédito hipotecario convencional."
        ),
        acciones=[
            f"{b('Microcrédito para mejora de vivienda:')} Líneas de crédito de hasta 50,000 MXN con tasa preferencial (máx. TIIE + 2%), plazos de 24–60 meses y sin aval.",
            f"{b('Esquemas de ahorro previo:')} Programas de ahorro colectivo (tandas formalizadas, ROSCA) que generen historial y capital semilla.",
            f"{b('Crédito puente progresivo:')} Financiamiento escalonado que avanza conforme se completa cada etapa constructiva, reduciendo riesgo moral.",
            f"{b('Banca de desarrollo:')} Fondeo a través de FOVISSSTE-Rural, Financiera Nacional de Desarrollo o fideicomisos estatales para intermediarios financieros locales.",
            f"{b('Garantías gubernamentales:')} Fondos de garantía que cubran riesgo de impago, permitiendo tasas menores a intermediarios privados.",
        ],
        actores=[
            "FOVISSSTE, SHF, Banca de Desarrollo (Banobras, FND)",
            "Cooperativas de ahorro y crédito, IMFs (instituciones de microfinanzas)",
            "CONDUSEF (educación y regulación)",
        ],
        mecanismo=(
            "En el ABM, el microcrédito se modela como un incremento en la tasa de "
            "ahorro (theta): theta_max sube hasta 25 pp y theta_min hasta 5 pp. "
            "El efecto es que los hogares pueden destinar una fracción mayor de su "
            "excedente de ingreso a construcción, acelerando la acumulación de capital."
        ),
        params=[
            ("theta_max (tasa ahorro máx.)", "+12.5 pp (ej. 0.60 → 0.73)", "+25 pp (máx. 0.95)"),
            ("theta_min (tasa ahorro mín.)", "+2.5 pp", "+5 pp"),
        ],
        impacto=(
            "Impacto moderado (Δ ≈ −2.0 años a 50% intensidad). Más efectivo que "
            "el subsidio cuando el hogar ya tiene ingreso pero le falta liquidez "
            "inmediata. Su efectividad depende de la regularidad del ingreso."
        ),
    ),
    # ─────────────────────────────────────────────────────────────────────────
    dict(
        codigo="C",
        nombre="Reducción de shocks económicos",
        color=POL_COLS["C"],
        objetivo=(
            "Proteger el patrimonio constructivo de los hogares frente a eventos "
            "adversos (pérdida de empleo, enfermedad, desastre natural) que destruyen "
            "el ahorro acumulado y revierten el avance en la vivienda."
        ),
        problema=(
            "Los shocks económicos son el principal factor de rezago habitacional: "
            "un solo evento puede eliminar años de ahorro y forzar al hogar a "
            "retroceder en su proceso constructivo. El quintil 1 carece de colchón "
            "financiero para absorber estos choques."
        ),
        acciones=[
            f"{b('Seguro de desempleo para trabajadores informales:')} Transferencias temporales ante pérdida de ingreso, financiadas con mecanismos de solidaridad comunitaria o fondos estatales.",
            f"{b('Seguro de salud universal efectivo:')} Eliminar el gasto catastrófico en salud que representa el shock económico más frecuente y severo para el quintil 1.",
            f"{b('Fondos de emergencia comunitarios:')} Cajas de ahorro o fondos municipales de emergencia con acceso inmediato para reparar daños o cubrir gastos imprevistos.",
            f"{b('Seguro paramétrico para vivienda:')} Productos de seguro ligados a índices objetivos (sismicidad, lluvias, etc.) que paguen automáticamente sin ajustador.",
            f"{b('Programa de reconstrucción rápida:')} Apoyo gubernamental predefinido ante declaratoria de emergencia, con materiales y mano de obra, para no perder avance constructivo.",
        ],
        actores=[
            "IMSS Bienestar, SSA (salud universal)",
            "FONDEN / CENAPRED (riesgo de desastres)",
            "Municipios (fondos comunitarios de emergencia)",
            "Aseguradoras con regulación de CNSF",
        ],
        mecanismo=(
            "En el ABM, los shocks reducen el ingreso en un porcentaje (shock_delta) "
            "con cierta probabilidad (shock_prob_base). La política reduce ambos "
            "parámetros: hasta 70% menos de probabilidad y 60% menos de severidad "
            "a máxima intensidad."
        ),
        params=[
            ("shock_prob_base", "−35% de probabilidad (0.10 → 0.065)", "−70% (0.10 → 0.030)"),
            ("shock_delta (severidad)", "−30% de impacto (0.30 → 0.21)", "−60% (0.30 → 0.12)"),
        ],
        impacto=(
            f"{b('Lever más poderoso del modelo')} (Δ ≈ −6.1 años a 50% intensidad). "
            "La reducción de shocks tiene el mayor impacto individual porque actúa "
            "sobre el factor que más destruye el ahorro acumulado. Fundamental para "
            "consolidar los avances de otras políticas."
        ),
    ),
    # ─────────────────────────────────────────────────────────────────────────
    dict(
        codigo="D",
        nombre="Densificación de oferentes",
        color=POL_COLS["D"],
        objetivo=(
            "Aumentar la disponibilidad de proveedores de materiales y servicios "
            "de construcción en comunidades de bajos ingresos, reduciendo costos "
            "por competencia e incrementando la probabilidad de encontrar mano "
            "de obra calificada."
        ),
        problema=(
            "En localidades marginadas, la escasez de proveedores formales encarece "
            "los materiales (transporte, intermediarios) y limita el acceso a "
            "técnicos competentes. La informalidad del mercado de construcción "
            "genera costos ocultos y obra de baja calidad."
        ),
        acciones=[
            f"{b('Capacitación de albañiles y técnicos locales:')} Programas de certificación de competencias laborales (CONOCER) orientados a autoconstrucción progresiva.",
            f"{b('Centros comunitarios de materiales:')} Puntos de venta de materiales subsidiados operados por cooperativas locales o municipios, eliminando intermediarios.",
            f"{b('Brigadas técnicas móviles:')} Equipos de arquitectos y técnicos que visitan colonias para asesoría en diseño, cálculo estructural y supervisión de obra.",
            f"{b('Plataforma digital de oferta:')} Marketplace local de contratistas, albañiles y proveedores con calificaciones y precios, reduciendo asimetría de información.",
            f"{b('Incentivos fiscales a proveedores:')} Exención de impuestos o apoyos a proveedores formales que abran sucursales en zonas de alta marginación.",
        ],
        actores=[
            "STPS / CONOCER (certificación laboral)",
            "Municipios y gobiernos estatales (centros de materiales)",
            "CONACYT / universidades (brigadas técnicas)",
            "Sector privado (constructoras, ferreterías con esquemas CSR)",
        ],
        mecanismo=(
            "En el ABM, la política aumenta la densidad de oferentes (dens_of_factor) "
            "hasta 5 veces la línea base. Mayor densidad reduce el factor de costo por "
            "oferta (supply_f) e incrementa la probabilidad de transición vía el "
            "término logístico alpha_3 × log(dens_oferentes)."
        ),
        params=[
            ("dens_of_factor", "×3 (triple de oferentes)", "×5 (cinco veces más)"),
            ("Costo (supply_f)", "−29% aprox.", "−50% (piso del modelo)"),
        ],
        impacto=(
            "Impacto significativo (Δ ≈ −5.1 años a 50% intensidad), especialmente "
            "en zonas rurales y de alta marginación donde la escasez de proveedores "
            "es más severa. Complementa al subsidio al reducir el costo unitario."
        ),
    ),
    # ─────────────────────────────────────────────────────────────────────────
    dict(
        codigo="E",
        nombre="Educación financiera y formalización",
        color=POL_COLS["E"],
        objetivo=(
            "Mejorar la capacidad de planeación financiera y la vinculación con "
            "mercados formales de trabajo y crédito, ampliando las opciones de "
            "ahorro e inversión de los hogares del quintil 1."
        ),
        problema=(
            "La falta de educación financiera genera decisiones subóptimas de ahorro "
            "e inversión. La informalidad laboral excluye a los hogares del sistema "
            "de protección social y financiero. El índice compuesto educación-formalidad "
            "del quintil 1 es el más bajo del espectro, limitando su acceso a "
            "instrumentos que podrían acelerar la autoproducción."
        ),
        acciones=[
            f"{b('Talleres de educación financiera:')} Módulos prácticos sobre presupuesto familiar, ahorro para construcción, uso de crédito y gestión de shocks, impartidos en centros comunitarios.",
            f"{b('Programas de formalización laboral:')} Incentivos para que empleadores registren a trabajadores en IMSS, ampliando acceso a crédito Infonavit y protección social.",
            f"{b('Cuenta de ahorro para vivienda:')} Producto financiero específico con rendimiento preferencial y retiro condicionado a uso en construcción.",
            f"{b('Asesoría para trámites de regularización:')} Apoyo para obtener escrituras y permisos de construcción, desbloqueando el acceso a programas gubernamentales.",
            f"{b('Vinculación con programa de becas técnicas:')} Apoyo para que miembros del hogar estudien carreras técnicas de construcción, aumentando capital humano disponible.",
        ],
        actores=[
            "SEP / INEA (educación de adultos)",
            "CONDUSEF (educación financiera)",
            "IMSS / STPS (formalización laboral)",
            "Banca comercial y de desarrollo (productos de ahorro)",
        ],
        mecanismo=(
            "En el ABM, la educación mejora el índice compuesto x_i del hogar "
            "(educación, formalidad financiera, marginación), que entra al logit "
            "de transición vía alpha_4. También se incrementa alpha_3 para reflejar "
            "mayor aprovechamiento de la oferta local de proveedores."
        ),
        params=[
            ("alpha_4 (efecto índice educ.)", "+0.25 (ej. 0.50 → 0.75)", "+0.50 (0.50 → 1.00)"),
            ("alpha_3 (efecto densidad)", "+0.15", "+0.30"),
        ],
        impacto=(
            "Impacto menor en el corto plazo (Δ ≈ −0.09 años a 50% intensidad), "
            "pero con efectos acumulativos que se profundizan en horizontes largos. "
            "Su mayor valor es complementario: amplifica el efecto de otras políticas "
            "al mejorar la capacidad de los hogares para aprovecharlas."
        ),
    ),
    # ─────────────────────────────────────────────────────────────────────────
    dict(
        codigo="F",
        nombre="Paquete integral (A + C + E)",
        color=POL_COLS["F"],
        objetivo=(
            "Atacar simultáneamente las tres barreras fundamentales de la "
            "autoproducción: costo de construcción (A), volatilidad económica (C) "
            "y capacidad de gestión financiera (E), generando efectos sinérgicos "
            "que ninguna política individual puede lograr por separado."
        ),
        problema=(
            "Las barreras a la autoproducción son sistémicas e interdependientes: "
            "un subsidio sin protección contra shocks se pierde en el primer evento "
            "adverso; la educación financiera sin acceso a capital no genera avance; "
            "la reducción de costos sin ahorro previo no activa la construcción. "
            "Solo un paquete integral rompe el ciclo de pobreza habitacional."
        ),
        acciones=[
            f"{b('Plataforma unificada de atención:')} Ventanilla única que otorga subsidio (A) + producto de ahorro/crédito (B) + seguro de emergencia (C) + asesoría financiera (E) en un solo trámite.",
            f"{b('Condicionalidad cruzada:')} El subsidio se libera por etapas, condicionado a participación en talleres de educación financiera y contratación del seguro de emergencia.",
            f"{b('Gestor comunitario de vivienda:')} Promotor capacitado que acompaña al hogar durante todo el proceso, coordinando el acceso a los distintos componentes del paquete.",
            f"{b('Presupuesto participativo para materiales:')} Compras consolidadas a nivel colonia que reducen costo unitario mediante economías de escala.",
            f"{b('Sistema de seguimiento digital:')} App o plataforma que registra el avance constructivo, libera transferencias automáticas y alerta sobre riesgo de shock o paralización.",
        ],
        actores=[
            "SEDATU (coordinación intersectorial)",
            "BIENESTAR / programas sociales federales",
            "Gobiernos estatales y municipales (implementación local)",
            "Organizaciones de la sociedad civil (gestores comunitarios)",
            "Banca de desarrollo (productos financieros vinculados)",
        ],
        mecanismo=(
            "En el ABM, el paquete integral aplica los componentes A, C y E a plena "
            "intensidad simultáneamente, generando el mayor impacto del modelo. "
            "El slider de intensidad controla el nivel de inversión; el diseño "
            "evita dobles penalizaciones para preservar la superioridad del paquete "
            "sobre cualquier política individual."
        ),
        params=[
            ("Costo E3 / otras etapas", "−20% / −14% (como política A)", "−40% / −28%"),
            ("shock_prob / shock_delta", "−35% / −30% (como política C)", "−70% / −60%"),
            ("alpha_4 / alpha_3", "+0.25 / +0.15 (como política E)", "+0.50 / +0.30"),
        ],
        impacto=(
            f"{b('Mayor impacto del portafolio')} (Δ ≈ −7.2 años a 50%, Δ ≈ −11.2 años a 100%). "
            "Supera a cualquier política individual a todas las intensidades. "
            "La sinergia entre reducción de shocks (que protege el ahorro) y subsidio "
            "(que reduce el costo objetivo) es el motor principal del efecto acumulado."
        ),
    ),
]


# ── Builder ───────────────────────────────────────────────────────────────────
def build_pdf(path):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
    )
    story = []

    # ── Portada ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph(
        "Políticas públicas para la autoproducción de vivienda",
        titulo_doc,
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Descripción, acciones e impacto simulado · Quintil 1 de ingresos · ENIGH 2024",
        subtitulo_doc,
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(hr(AZUL))
    story.append(Spacer(1, 0.3*cm))

    # Contexto
    story.append(Paragraph("Contexto del modelo", subseccion))
    story.append(Paragraph(
        "Este documento describe las seis políticas públicas implementadas en el "
        "Modelo Basado en Agentes (ABM) de autoproducción de vivienda para hogares "
        "del quintil 1 de ingresos en México (ENIGH 2024). El ABM simula el proceso "
        "de construcción progresiva a través de cuatro etapas: "
        f"{b('E1 Inicial')} → {b('E2 Consolidación')} → {b('E3 Terminaciones')} → "
        f"{b('E4 Servicios básicos')}. El diagnóstico base revela dos cuellos de botella "
        "principales: el rezago extremo en E1 (sin excedente de ingreso no hay ahorro) "
        "y la transición E3→E4 (servicios básicos requieren infraestructura externa). "
        "Las políticas simuladas atacan estos bloqueos desde distintos ángulos.",
        cuerpo,
    ))

    # Tabla resumen de impactos
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Resumen comparativo de impacto (T = 20 años, intensidad 50%)", subseccion))
    impact_data = [
        [Paragraph(b("Política"), param_label),
         Paragraph(b("Años E1→E4 (base: 25.9)"), param_label),
         Paragraph(b("Δ años"), param_label),
         Paragraph(b("Lever principal"), param_label)],
        ["A · Subsidio",       "24.6 años", "−1.3", "Reducción de costo"],
        ["B · Microcrédito",   "23.8 años", "−2.0", "Incremento de ahorro"],
        ["C · Shocks",         "19.8 años", "−6.1", "Protección patrimonial"],
        ["D · Oferentes",      "20.8 años", "−5.1", "Acceso a proveedores"],
        ["E · Educación",      "25.8 años", "−0.1", "Capital humano (L/P)"],
        ["F · Integral A+C+E", "18.7 años", "−7.2", "Sinergia sistémica"],
    ]
    col_colors = [ROSA, MORADO, NARANJA2, VERDE2, AZUL2, VERDE3]
    impact_tbl = Table(impact_data, colWidths=[4.5*cm, 4*cm, 2*cm, 6.2*cm])
    ts = [
        ("BACKGROUND",    (0,0), (-1,0), NEGRO),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 7),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#DDDDDD")),
        ("ALIGN",         (2,1), (2,-1), "CENTER"),
    ]
    for i_row, col in enumerate(col_colors, start=1):
        ts.append(("BACKGROUND", (0, i_row), (0, i_row), col))
        ts.append(("TEXTCOLOR",  (0, i_row), (0, i_row), colors.white))
        ts.append(("FONTNAME",   (0, i_row), (0, i_row), "Helvetica-Bold"))
        ts.append(("TEXTCOLOR",  (2, i_row), (2, i_row), col))
        ts.append(("FONTNAME",   (2, i_row), (2, i_row), "Helvetica-Bold"))
    impact_tbl.setStyle(TableStyle(ts))
    story.append(impact_tbl)
    story.append(Paragraph(
        "Nota: Los años esperados para alcanzar E4 se calculan con la fórmula de "
        "tiempo de absorción de cadenas de Markov, anclada en P_inicial (ENIGH 2024) "
        "y escalada por la mejora relativa observada en el ABM.",
        nota,
    ))

    # ── Páginas por política ──────────────────────────────────────────────────
    for pol in POLITICAS:
        story.append(PageBreak())
        block = []

        # Encabezado
        block.append(header_pol(pol["codigo"], pol["nombre"], pol["color"]))
        block.append(Spacer(1, 0.4*cm))

        # Objetivo
        block.append(Paragraph("Objetivo de política", subseccion))
        block.append(Paragraph(pol["objetivo"], cuerpo))

        # Problema que atiende
        block.append(Paragraph("Problema que atiende", subseccion))
        block.append(Paragraph(pol["problema"], cuerpo))

        # Acciones
        block.append(Paragraph("Acciones e instrumentos", subseccion))
        for a in pol["acciones"]:
            block.append(bullet(a))
        block.append(Spacer(1, 0.1*cm))

        # Actores
        block.append(Paragraph("Actores clave", subseccion))
        for a in pol["actores"]:
            block.append(bullet(i(a)))
        block.append(Spacer(1, 0.1*cm))

        # Modelado
        block.append(Paragraph("Modelado en el ABM", subseccion))
        block.append(Paragraph(pol["mecanismo"], cuerpo))
        block.append(Spacer(1, 0.2*cm))
        block.append(param_table(pol["params"], pol["color"]))
        block.append(Spacer(1, 0.2*cm))

        # Impacto
        block.append(Paragraph("Impacto esperado", subseccion))
        block.append(Paragraph(pol["impacto"], cuerpo))

        story.append(KeepTogether(block[:6]))
        story.extend(block[6:])

    # ── Página final: consideraciones ────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Consideraciones metodológicas", subseccion))
    story.append(hr())

    consideraciones = [
        (
            "Escenarios macro",
            "Las simulaciones contemplan tres escenarios macroeconómicos: "
            f"{b('Conservador')} (menor inflación y shocks), {b('Base')} (parámetros ENIGH 2024) "
            f"y {b('Crítico')} (mayor inflación de construcción y volatilidad). "
            "Los impactos reportados corresponden al escenario Base."
        ),
        (
            "Horizonte temporal",
            "El dashboard permite simular horizontes de 5 a 40 años. Los valores "
            "de impacto citados en este documento corresponden a un horizonte de "
            "20 años, que es el plazo en que la mayoría de hogares del quintil 1 "
            "puede aspirar a completar su proceso constructivo con apoyo institucional."
        ),
        (
            "Interacción entre políticas",
            "El paquete integral (F) es estrictamente superior a cualquier política "
            "individual a toda intensidad. Las políticas C (shocks) y D (oferentes) "
            "son los levers más poderosos. La política E (educación) tiene el mayor "
            "impacto de largo plazo y amplifica las otras al mejorar la capacidad "
            "de aprovechamiento del hogar."
        ),
        (
            "Limitaciones del modelo",
            "El ABM abstrae heterogeneidades importantes: geografía, tipo de tenencia, "
            "composición del hogar y acceso diferenciado a servicios. Los resultados "
            "deben interpretarse como órdenes de magnitud comparativos, no como "
            "proyecciones exactas. La calibración con datos ENIGH 2024 representa "
            "el promedio del quintil 1 nacional."
        ),
    ]
    for titulo, texto in consideraciones:
        story.append(Paragraph(b(titulo), cuerpo))
        story.append(Paragraph(texto, cuerpo))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(hr(GRIS))
    story.append(Paragraph(
        "Fuente: Elaboración propia — ABM con datos ENIGH 2024, INEGI. "
        "Transiciones modeladas con función logística condicionada por cadena de Markov empírica. "
        "Simulador disponible en dashboard_politicas.py.",
        caption,
    ))

    doc.build(story)
    print(f"PDF generado: {path}")


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "politicas_vivienda_ABM.pdf")
    build_pdf(out)
