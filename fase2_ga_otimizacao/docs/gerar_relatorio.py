"""
Gerador de Relatório Técnico PDF — Tech Challenge Fase 2
FIAP Pós Tech — Algoritmos Genéticos e LLMs
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus.flowables import Flowable

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\guivd\Desktop\python projects\Tech_Challenge_Fase1\fase2_ga_otimizacao"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
OUTPUT_PDF = os.path.join(DOCS_DIR, "Relatorio_Tecnico_Fase2.pdf")

IMG_CONVERGENCE = os.path.join(RESULTS_DIR, "convergence.png")
IMG_METRICS = os.path.join(RESULTS_DIR, "metrics_comparison.png")

# ─── Colors ─────────────────────────────────────────────────────────────────
NAVY = HexColor('#1a237e')
RED = HexColor('#c62828')
LIGHT_BLUE = HexColor('#e3f2fd')
MID_BLUE = HexColor('#bbdefb')
DARK_GRAY = HexColor('#37474f')
LIGHT_GRAY = HexColor('#f5f5f5')
WHITE = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm


# ─── Page Template with header/footer ───────────────────────────────────────
def make_canvas_decorator(doc):
    def on_page(canvas, doc):
        canvas.saveState()
        # Top color bar
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 1.0 * cm, PAGE_W, 1.0 * cm, fill=1, stroke=0)
        # Top bar text
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(MARGIN, PAGE_H - 0.65 * cm,
                          "FIAP Pos Tech  |  Tech Challenge Fase 2  |  Algoritmos Geneticos e LLMs")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.65 * cm,
                               "Otimizacao via Algoritmo Genetico")
        # Bottom bar
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_W, 0.8 * cm, fill=1, stroke=0)
        # Page number
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 7)
        canvas.drawCentredString(PAGE_W / 2, 0.25 * cm,
                                 f"Pagina {doc.page}")
        # Red accent line
        canvas.setFillColor(RED)
        canvas.rect(0, PAGE_H - 1.0 * cm - 0.12 * cm, PAGE_W, 0.12 * cm, fill=1, stroke=0)
        canvas.restoreState()

    def on_first_page(canvas, doc):
        canvas.saveState()
        # Full navy background for cover
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.restoreState()

    return on_first_page, on_page


# ─── Styles ─────────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles['cover_title'] = ParagraphStyle(
        'cover_title',
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=WHITE,
        alignment=TA_CENTER,
        leading=28,
        spaceAfter=10,
    )
    styles['cover_subtitle'] = ParagraphStyle(
        'cover_subtitle',
        fontName='Helvetica',
        fontSize=13,
        textColor=HexColor('#e3f2fd'),
        alignment=TA_CENTER,
        leading=18,
        spaceAfter=6,
    )
    styles['cover_meta'] = ParagraphStyle(
        'cover_meta',
        fontName='Helvetica',
        fontSize=10,
        textColor=HexColor('#90caf9'),
        alignment=TA_CENTER,
        leading=15,
        spaceAfter=4,
    )
    styles['cover_meta_bold'] = ParagraphStyle(
        'cover_meta_bold',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=WHITE,
        alignment=TA_CENTER,
        leading=15,
        spaceAfter=4,
    )
    styles['section_title'] = ParagraphStyle(
        'section_title',
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=WHITE,
        alignment=TA_LEFT,
        leading=16,
        spaceBefore=14,
        spaceAfter=8,
        leftIndent=0,
    )
    styles['subsection_title'] = ParagraphStyle(
        'subsection_title',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=NAVY,
        alignment=TA_LEFT,
        leading=14,
        spaceBefore=10,
        spaceAfter=5,
        leftIndent=0,
    )
    styles['body'] = ParagraphStyle(
        'body',
        fontName='Helvetica',
        fontSize=9.5,
        textColor=DARK_GRAY,
        alignment=TA_JUSTIFY,
        leading=14,
        spaceAfter=5,
    )
    styles['body_bold'] = ParagraphStyle(
        'body_bold',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=DARK_GRAY,
        alignment=TA_LEFT,
        leading=14,
        spaceAfter=4,
    )
    styles['bullet'] = ParagraphStyle(
        'bullet',
        fontName='Helvetica',
        fontSize=9.5,
        textColor=DARK_GRAY,
        alignment=TA_LEFT,
        leading=14,
        spaceAfter=3,
        leftIndent=14,
        bulletIndent=4,
    )
    styles['caption'] = ParagraphStyle(
        'caption',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        textColor=HexColor('#546e7a'),
        alignment=TA_CENTER,
        leading=12,
        spaceBefore=4,
        spaceAfter=8,
    )
    styles['code'] = ParagraphStyle(
        'code',
        fontName='Helvetica',
        fontSize=9,
        textColor=HexColor('#1a237e'),
        backColor=HexColor('#e8eaf6'),
        alignment=TA_LEFT,
        leading=13,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=10,
        rightIndent=10,
        borderPad=6,
    )
    return styles


# ─── Helpers ────────────────────────────────────────────────────────────────

def section_header(title, styles):
    """Returns a navy background section header."""
    tbl = Table([[Paragraph(title, styles['section_title'])]], colWidths=[PAGE_W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    return tbl


def build_table(headers, rows, styles, col_widths=None):
    """Build a styled table with alternating row colors."""
    available = PAGE_W - 2 * MARGIN
    if col_widths is None:
        n = len(headers)
        col_widths = [available / n] * n

    header_style = ParagraphStyle(
        'th', fontName='Helvetica-Bold', fontSize=8.5,
        textColor=WHITE, alignment=TA_CENTER, leading=12
    )
    cell_style = ParagraphStyle(
        'td', fontName='Helvetica', fontSize=8.5,
        textColor=DARK_GRAY, alignment=TA_LEFT, leading=12
    )

    data = [[Paragraph(h, header_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), cell_style) for c in row])

    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    ts = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.3, HexColor('#b0bec5')),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, WHITE]),
    ]
    tbl.setStyle(TableStyle(ts))
    return tbl


def insert_image(path, caption_text, styles, max_width=None, max_height=None):
    """Return list of flowables: image + caption, or notice if missing."""
    if max_width is None:
        max_width = PAGE_W - 2 * MARGIN - 1 * cm
    if max_height is None:
        max_height = 9 * cm

    elements = []
    if os.path.exists(path):
        img = Image(path)
        # Scale proportionally
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_width / iw, max_height / ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale

        img_tbl = Table([[img]], colWidths=[PAGE_W - 2 * MARGIN])
        img_tbl.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
            ('BOX', (0, 0), (-1, -1), 0.5, HexColor('#b0bec5')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(img_tbl)
        elements.append(Paragraph(caption_text, styles['caption']))
    else:
        elements.append(Paragraph(f"[Imagem nao encontrada: {os.path.basename(path)}]",
                                  styles['caption']))
    return elements


# ─── Cover Page ─────────────────────────────────────────────────────────────
def cover_page(styles):
    elements = []
    # Large top spacer (cover has navy background via canvas)
    elements.append(Spacer(1, 5.5 * cm))

    # Red accent bar
    bar = Table([['']], colWidths=[8 * cm], rowHeights=[0.35 * cm])
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), RED),
    ]))
    bar_wrap = Table([[bar]], colWidths=[PAGE_W - 2 * MARGIN])
    bar_wrap.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    elements.append(bar_wrap)
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph("Relatorio Tecnico", styles['cover_subtitle']))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph("Tech Challenge Fase 2", styles['cover_title']))
    elements.append(Spacer(1, 0.4 * cm))

    subtitle_text = (
        "Otimizacao de Modelos de Diagnostico para Saude da Mulher<br/>"
        "via Algoritmo Genetico + LLM"
    )
    elements.append(Paragraph(subtitle_text, styles['cover_subtitle']))
    elements.append(Spacer(1, 1.5 * cm))

    # Divider line
    div = Table([['']], colWidths=[12 * cm], rowHeights=[0.08 * cm])
    div.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor('#3949ab'))]))
    div_wrap = Table([[div]], colWidths=[PAGE_W - 2 * MARGIN])
    div_wrap.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    elements.append(div_wrap)
    elements.append(Spacer(1, 1.0 * cm))

    meta_items = [
        ("Instituicao", "FIAP  -  Pos Tech"),
        ("Fase", "2  -  Algoritmos Geneticos e LLMs"),
        ("Projeto", "Projeto 1  -  Otimizacao de Modelos de Diagnostico"),
        ("Dataset", "Breast Cancer Wisconsin (UCI)"),
        ("Data", "Marco 2026"),
    ]
    for label, value in meta_items:
        row_tbl = Table(
            [[Paragraph(f"{label}:", styles['cover_meta']),
              Paragraph(value, styles['cover_meta_bold'])]],
            colWidths=[5 * cm, 10 * cm]
        )
        row_tbl.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
            ('ALIGN', (1, 0), (1, 0), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        wrap = Table([[row_tbl]], colWidths=[PAGE_W - 2 * MARGIN])
        wrap.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        elements.append(wrap)

    elements.append(Spacer(1, 2 * cm))

    # Bottom red bar on cover
    bottom_bar = Table([['']], colWidths=[PAGE_W - 2 * MARGIN], rowHeights=[0.15 * cm])
    bottom_bar.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), RED)]))
    elements.append(bottom_bar)

    elements.append(PageBreak())
    return elements


# ─── Build full document ────────────────────────────────────────────────────
def build_document():
    styles = build_styles()

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=1.8 * cm,
        bottomMargin=1.5 * cm,
        title="Relatorio Tecnico - Tech Challenge Fase 2",
        author="FIAP Pos Tech",
        subject="Otimizacao via Algoritmo Genetico + LLM",
    )

    on_first, on_later = make_canvas_decorator(doc)

    story = []

    # ── COVER ──────────────────────────────────────────────────────────────
    story.extend(cover_page(styles))

    # ── SECTION 1 ──────────────────────────────────────────────────────────
    story.append(section_header("1. Introducao e Contexto", styles))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "O objetivo deste projeto e otimizar automaticamente os hiperparametros do modelo de "
        "diagnostico de cancer de mama desenvolvido na Fase 1, utilizando Algoritmo Genetico (AG). "
        "Adicionalmente, integramos um LLM local (flan-t5-base) para transformar os resultados "
        "numericos em linguagem clinica acessivel aos profissionais de saude.",
        styles['body']
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "O modelo da Fase 1 utilizava Regressao Logistica com hiperparametros definidos manualmente:",
        styles['body']
    ))
    for item in [
        "Scaler: RobustScaler",
        "PCA: 90% de variancia",
        "C: 1.0 (regularizacao padrao)",
        "Solver: lbfgs",
    ]:
        story.append(Paragraph(f"  -  {item}", styles['bullet']))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "A hipotese central e que o AG pode encontrar uma combinacao superior de hiperparametros "
        "ao explorar o espaco de busca de forma inteligente, sem a necessidade de grid search exaustivo.",
        styles['body']
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── SECTION 2 ──────────────────────────────────────────────────────────
    story.append(section_header("2. Implementacao do Algoritmo Genetico", styles))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2.1 Representacao dos Genes (Cromossomo)", styles['subsection_title']))
    story.append(Paragraph(
        "Adotamos codificacao real (real-valued encoding): cada individuo e um vetor de 6 genes "
        "reais em [0, 1].",
        styles['body']
    ))
    story.append(Spacer(1, 0.2 * cm))

    gene_headers = ["Gene", "Hiperparametro", "Mapeamento"]
    gene_rows = [
        ["0", "Scaler", "Categorico: StandardScaler / RobustScaler / MinMaxScaler"],
        ["1", "Usar PCA", "Binario: >= 0.5 = True"],
        ["2", "PCA variance", "Continuo: 0.80 - 0.99"],
        ["3", "C (LogReg)", "Log-escala: 10^(-3 a +2) = 0.001 a 100"],
        ["4", "Solver", "Categorico: lbfgs / liblinear / saga"],
        ["5", "max_iter", "Inteiro: 200 - 2000"],
    ]
    story.append(build_table(gene_headers, gene_rows, styles,
                             col_widths=[1.5 * cm, 3.5 * cm, PAGE_W - 2 * MARGIN - 5 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "Justificativa: A codificacao real e superior a binaria para hiperparametros continuos "
        "como C, que varia em ordens de magnitude. O crossover e a mutacao operam diretamente "
        "nos valores reais, permitindo refinamento fino do espaco de busca.",
        styles['body']
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2.2 Funcao Fitness", styles['subsection_title']))
    story.append(Paragraph(
        "fitness = 0.45 x Recall + 0.25 x F1-score + 0.20 x Specificity + 0.10 x Equity",
        styles['code']
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Justificativa dos pesos:", styles['body_bold']))
    justificativas = [
        ("Recall (45%)", "No contexto oncologico, false negatives (cancer nao detectado) sao "
         "clinicamente mais graves que false positives. Uma paciente com diagnostico perdido pode "
         "perder a janela de tratamento curativo."),
        ("F1-score (25%)", "Garante equilibrio geral entre precision e recall, evitando que o "
         "modelo maximize recall as custas de specificity extremamente baixa."),
        ("Specificity (20%)", "Controla os false positives - pacientes saudaveis diagnosticadas "
         "erroneamente geram ansiedade, biopsias desnecessarias e sobrecarga do sistema de saude."),
        ("Equity (10%)", "Mede a consistencia do recall entre grupos demograficos (quartis de mean "
         "radius). Um modelo equitativo tem desempenho consistente independentemente do perfil da "
         "paciente."),
    ]
    for term, desc in justificativas:
        story.append(Paragraph(f"  -  <b>{term}:</b> {desc}", styles['bullet']))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "A avaliacao utiliza stratified k-fold cross-validation (5 folds) para garantir generalizacao.",
        styles['body']
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2.3 Operadores Geneticos", styles['subsection_title']))

    operators = [
        ("Selection - Tournament Selection (k=3)",
         "Tres individuos sao sorteados aleatoriamente; o de maior fitness e selecionado para "
         "reproducao. Cria pressao seletiva consistente independente da escala absoluta dos "
         "valores de fitness - essencial quando a populacao ja esta bem convergida "
         "(fitness entre 0.96 e 0.98)."),
        ("Crossover - Uniform e Arithmetic (alternados)",
         "Uniform Crossover: cada gene e herdado de um dos pais com probabilidade 0.5. Adequado "
         "para genes independentes. Arithmetic Crossover: filho = alpha x pai1 + (1-alpha) x pai2, "
         "com alpha entre 0.3 e 0.7. Ideal para genes continuos (C, PCA variance), gerando filhos "
         "com valores intermediarios."),
        ("Mutation - Gaussian Mutation",
         "Cada gene sofre perturbacao N(0, sigma) com probabilidade mutation_rate. A distribuicao "
         "gaussiana realiza pequenas perturbacoes ao redor do valor atual, ideal para refinamento "
         "fino. Apos a perturbacao, os genes sao recortados em [0, 1]."),
        ("Elitism",
         "Os 2 melhores individuos de cada geracao passam diretamente para a proxima, garantindo "
         "que o melhor modelo encontrado nunca seja perdido."),
    ]
    for op_name, op_desc in operators:
        story.append(Paragraph(f"<b>{op_name}:</b>", styles['body_bold']))
        story.append(Paragraph(op_desc, styles['body']))
        story.append(Spacer(1, 0.15 * cm))

    story.append(Spacer(1, 0.3 * cm))

    # ── SECTION 3 ──────────────────────────────────────────────────────────
    story.append(section_header("3. Experimentos Realizados", styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Foram realizados 3 experimentos com diferentes configuracoes do AG:",
        styles['body']
    ))
    story.append(Spacer(1, 0.2 * cm))

    exp_headers = ["Experimento", "Pop", "Mutation Rate", "Geracoes", "Objetivo"]
    exp_rows = [
        ["Exp 1", "30", "0.15", "20", "Fast convergence (comparison baseline)"],
        ["Exp 2", "50", "0.10", "30", "Exploration/exploitation balance"],
        ["Exp 3", "80", "0.20", "30", "High diversity - maior exploration do espaco"],
    ]
    w = PAGE_W - 2 * MARGIN
    story.append(build_table(exp_headers, exp_rows, styles,
                             col_widths=[2.5 * cm, 1.5 * cm, 2.5 * cm, 2.5 * cm, w - 9 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("3.1 Resultados por Experimento", styles['subsection_title']))
    res_headers = ["Experimento", "Fitness Final", "Tempo", "Hiperparametros Otimos"]
    res_rows = [
        ["Exp 1", "0.9707", "27.8s", "StandardScaler | sem PCA | C=0.0961 | lbfgs"],
        ["Exp 2", "0.9714", "65.9s", "StandardScaler | sem PCA | C=0.1673 | liblinear"],
        ["Exp 3 (VENCEDOR)", "0.9730", "157.4s", "StandardScaler | PCA(0.98) | C=0.2706 | lbfgs"],
    ]
    story.append(build_table(res_headers, res_rows, styles,
                             col_widths=[3.5 * cm, 2.5 * cm, 2.0 * cm, w - 8 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("3.2 Analise dos Experimentos", styles['subsection_title']))
    story.append(Paragraph(
        "O Experimento 3 obteve o melhor fitness por dois fatores:",
        styles['body']
    ))
    story.append(Paragraph(
        "  1.  Larger population (80): Maior genetic diversity inicial, reduzindo o risco de "
        "premature convergence em local optima.",
        styles['bullet']
    ))
    story.append(Paragraph(
        "  2.  Higher mutation rate (0.20): Maior capacidade de exploration - o AG conseguiu "
        "sair de regioes subotimas onde os experimentos menores ficaram presos.",
        styles['bullet']
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "O trade-off e o tempo de execucao (157s vs 27s do Exp 1), aceitavel para um processo "
        "de otimizacao periodica offline.",
        styles['body']
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Image: convergence
    story.extend(insert_image(
        IMG_CONVERGENCE,
        "Figura 1 - Curvas de Convergencia dos 3 Experimentos do Algoritmo Genetico",
        styles
    ))
    story.append(Spacer(1, 0.3 * cm))

    # ── SECTION 4 ──────────────────────────────────────────────────────────
    story.append(section_header("4. Comparativo de Desempenho", styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Avaliacao no conjunto de teste (20% dos dados, 114 pacientes):",
        styles['body']
    ))
    story.append(Spacer(1, 0.2 * cm))

    perf_headers = ["Metrica", "Baseline (Fase 1)", "Otimizado (AG - Exp 3)", "Melhoria"]
    perf_rows = [
        ["Recall", "0.9524", "0.9762", "+0.0238"],
        ["F1-score", "0.9524", "0.9762", "+0.0238"],
        ["Accuracy", "0.9649", "0.9825", "+0.0176"],
        ["Specificity", "-", "0.9861", "-"],
        ["Precision (Malignant)", "0.95", "0.98", "+0.03"],
        ["Precision (Benign)", "0.97", "0.99", "+0.02"],
    ]
    story.append(build_table(perf_headers, perf_rows, styles,
                             col_widths=[3.5 * cm, 3.5 * cm, 4.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    # Image: metrics comparison
    story.extend(insert_image(
        IMG_METRICS,
        "Figura 2 - Comparativo de Metricas: Baseline (Fase 1) vs Modelo Otimizado (Exp 3)",
        styles
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4.1 Descobertas do AG", styles['subsection_title']))
    disc_headers = ["Hiperparametro", "Fase 1", "AG encontrou", "Interpretacao"]
    disc_rows = [
        ["Scaler", "RobustScaler", "StandardScaler",
         "Dataset nao tem outliers extremos que justifiquem RobustScaler"],
        ["PCA", "90% variancia", "98% variancia",
         "Manter mais componentes preserva informacao diagnostica"],
        ["C", "1.0", "0.27",
         "Regularizacao mais forte - modelo Fase 1 estava sub-regularizado"],
        ["Solver", "lbfgs", "lbfgs",
         "Correto para regularizacao L2 - AG confirmou"],
    ]
    story.append(build_table(disc_headers, disc_rows, styles,
                             col_widths=[2.5 * cm, 2.5 * cm, 2.5 * cm, w - 7.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4.2 Impacto Clinico", styles['subsection_title']))
    story.append(Paragraph(
        "A melhoria de +0.0238 no recall significa que, em um conjunto de 42 pacientes malignas "
        "(como no conjunto de teste), o modelo otimizado detecta 1 caso a mais que passaria "
        "despercebido pelo modelo original. Em escala hospitalar de 1.000 pacientes/mes, isso "
        "representa aproximadamente 24 mulheres diagnosticadas a mais por mes.",
        styles['body']
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4.3 Analise de Equidade Demografica", styles['subsection_title']))
    story.append(Paragraph(
        "A metrica de Equity na funcao fitness avalia a consistencia do recall entre diferentes "
        "subgrupos de pacientes, estratificados por quartis de mean_radius (proxy demografico). "
        "O modelo otimizado apresenta recall consistente entre os 4 quartis, indicando que a "
        "melhoria na deteccao beneficia igualmente pacientes com diferentes perfis tumorais - "
        "um requisito critico para equidade no diagnostico oncologico feminino.",
        styles['body']
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── SECTION 5 ──────────────────────────────────────────────────────────
    story.append(section_header("5. Integracao com LLM", styles))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("5.1 Modelo Escolhido", styles['subsection_title']))
    story.append(Paragraph(
        "google/flan-t5-base - modelo seq2seq de 250M parametros (~990 MB), executado 100% "
        "localmente sem necessidade de API ou internet apos o primeiro download.",
        styles['body']
    ))
    story.append(Paragraph("Justificativa da escolha local:", styles['body_bold']))
    for item in [
        "Privacidade: Dados de pacientes nao saem da instituicao",
        "Custo zero: Sem dependencia de APIs pagas",
        "Disponibilidade: Funciona offline - hospitais nao podem depender de internet",
    ]:
        story.append(Paragraph(f"  -  {item}", styles['bullet']))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("5.2 Tecnicas de Prompt Engineering", styles['subsection_title']))
    story.append(Paragraph("Todos os prompts foram projetados com 3 principios:", styles['body']))
    for item in [
        "Contexto medico feminino: especializacao em oncologia mamaria",
        "Sensibilidade de genero: linguagem empatica e acolhedora",
        "Privacidade: nenhum dado identificavel e enviado ao modelo",
    ]:
        story.append(Paragraph(f"  -  {item}", styles['bullet']))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "Os prompts foram otimizados para o modelo flan-t5 (seq2seq), usando instrucoes curtas "
        "e diretas que maximizam a qualidade da geracao. Um sistema de fallback baseado em "
        "templates estruturados garante respostas uteis mesmo quando o modelo nao gera texto adequado.",
        styles['body']
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Exemplo de prompt utilizado (explicacao de diagnostico):", styles['body_bold']))
    story.append(Paragraph(
        '"Medical diagnosis report for breast cancer. Result: malignant. Confidence: 87%. '
        'Recall: 97.6%. Write clinical explanation in Portuguese."',
        styles['code']
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("5.3 Outputs Gerados pelo LLM", styles['subsection_title']))
    out_headers = ["Output", "Publico-alvo", "Conteudo"]
    out_rows = [
        ["explicacao_diagnostico.txt", "Medico / Paciente",
         "Resultado, confianca, proximos passos clinicos"],
        ["comparacao_modelos.txt", "Gestao hospitalar",
         "Impacto clinico da melhoria, recomendacao de adocao"],
        ["analise_experimentos.txt", "Time tecnico",
         "Melhor configuracao do AG, analise comparativa"],
    ]
    story.append(build_table(out_headers, out_rows, styles,
                             col_widths=[5 * cm, 3.5 * cm, w - 8.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("5.4 Avaliacao de Qualidade das Respostas LLM", styles['subsection_title']))
    story.append(Paragraph(
        "Cada resposta e avaliada automaticamente em 4 criterios:",
        styles['body']
    ))
    story.append(Spacer(1, 0.15 * cm))
    eval_headers = ["Criterio", "Peso", "Metrica"]
    eval_rows = [
        ["Completeness", "30%", "Minimo de 150 palavras"],
        ["Medical Terminology", "40%", "Presenca de termos clinicos relevantes"],
        ["Adequacy", "20%", "Ausencia de linguagem alarmista"],
        ["Language (PT-BR)", "10%", "Marcadores do portugues brasileiro"],
    ]
    story.append(build_table(eval_headers, eval_rows, styles,
                             col_widths=[4 * cm, 2 * cm, w - 6 * cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        'As respostas avaliadas como "Boa" (score >= 0.7) sao salvas em llm_responses/*.json no '
        "formato prompt+resposta, constituindo o dataset inicial para fine-tuning na Fase 3.",
        styles['body']
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── SECTION 6 ──────────────────────────────────────────────────────────
    story.append(section_header("6. Arquitetura da Solucao", styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "O sistema e composto por 3 modulos principais:",
        styles['body']
    ))
    story.append(Spacer(1, 0.2 * cm))

    arch_items = [
        ("genetic_algorithm.py - Motor do AG",
         "Cromossomo com 6 genes reais [0,1] | Avaliacao por stratified k-fold CV (5 folds) | "
         "Operadores: Tournament Selection, Uniform/Arithmetic Crossover, Gaussian Mutation, Elitism"),
        ("llm.py - Modulo LLM Local",
         "Pipeline flan-t5-base (Hugging Face Transformers) | Deteccao de echo + fallback estruturado | "
         "Avaliacao automatica de qualidade | Persistencia das respostas para Fase 3"),
        ("main.py - Pipeline de Orquestracao",
         "Carregamento de dados, avaliacao baseline, execucao dos 3 experimentos | "
         "Geracao de graficos (convergence.png, metrics_comparison.png) | "
         "Exportacao de resultados em JSON"),
    ]
    for mod_name, mod_desc in arch_items:
        story.append(KeepTogether([
            Paragraph(f"<b>{mod_name}</b>", styles['body_bold']),
            Paragraph(mod_desc, styles['body']),
            Spacer(1, 0.15 * cm),
        ]))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Fluxo do Pipeline:", styles['body_bold']))
    story.append(Paragraph(
        "breast_cancer_dataset.csv  ->  load_data()  ->  Algoritmo Genetico (3 experimentos)  "
        "->  Avaliacao no conjunto de teste  ->  LLM Local (flan-t5-base)  "
        "->  llm_responses/*.json (dataset para Fase 3)",
        styles['code']
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── SECTION 7 ──────────────────────────────────────────────────────────
    story.append(section_header("7. Escalabilidade e Monitoramento", styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "O sistema foi projetado para execucao periodica offline, sem necessidade de "
        "infraestrutura em nuvem:",
        styles['body']
    ))
    for item in [
        "Cache de dados: O dataset e carregado uma unica vez em memoria",
        "Singleton do modelo LLM: O pipeline flan-t5 e inicializado uma vez e reutilizado",
        "Parametros configuráveis via CLI: --cv-folds permite balancear velocidade vs. precisao",
        "Logs estruturados: Todos os resultados sao salvos em JSON para auditoria e "
        "reprodutibilidade (experiment_results.json, llm_responses/*.json)",
    ]:
        story.append(Paragraph(f"  -  {item}", styles['bullet']))
    story.append(Spacer(1, 0.4 * cm))

    # ── SECTION 8 ──────────────────────────────────────────────────────────
    story.append(section_header("8. Desafios e Solucoes", styles))
    story.append(Spacer(1, 0.3 * cm))

    chall_headers = ["Desafio", "Solucao Implementada"]
    chall_rows = [
        ["Espaco em disco limitado para PyTorch",
         "Instalacao CPU-only (--index-url .../whl/cpu)"],
        ["flan-t5 ecoando prompts longos",
         "Prompts mais curtos + deteccao de echo (_is_echo()) + fallback com templates"],
        ["Avaliacao de equidade sem dados demograficos",
         "Proxy por quartis de mean_radius (feature 0 do dataset)"],
        ["Tempo de avaliacao do AG elevado",
         "--cv-folds 3 para desenvolvimento; 5 folds para resultados finais"],
        ["Hiperparametros em escalas muito diferentes",
         "Log-escala para C (0.001 a 100); codificacao real normalizada [0,1]"],
    ]
    story.append(build_table(chall_headers, chall_rows, styles,
                             col_widths=[5.5 * cm, w - 5.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    # ── SECTION 9 ──────────────────────────────────────────────────────────
    story.append(section_header("9. Consideracoes Eticas", styles))
    story.append(Spacer(1, 0.3 * cm))

    ethics_items = [
        ("Privacidade", "Nenhum dado identificavel de pacientes e utilizado ou transmitido. "
         "O LLM roda 100% local."),
        ("Equidade", "A funcao fitness inclui explicitamente uma metrica de equidade demografica "
         "(10% do peso), penalizando modelos com desempenho inconsistente entre subgrupos."),
        ("Vies (Bias)", "O uso de stratified k-fold garante representacao proporcional de classes "
         "maligno/benigno em todos os folds de avaliacao."),
        ("Responsabilidade", "O sistema e explicitamente apresentado como ferramenta de apoio a "
         "decisao - a decisao diagnostica final e sempre do medico."),
        ("Transparencia", "Todas as decisoes de design, pesos da fitness e limitacoes do modelo "
         "sao documentadas."),
        ("LGPD", "Os dados sinteticos/anonimos do dataset Wisconsin (UCI) nao contem informacoes "
         "pessoais identificaveis."),
    ]
    for term, desc in ethics_items:
        story.append(Paragraph(f"  -  <b>{term}:</b> {desc}", styles['bullet']))
        story.append(Spacer(1, 0.1 * cm))
    story.append(Spacer(1, 0.3 * cm))

    # ── SECTION 10 ─────────────────────────────────────────────────────────
    story.append(section_header("10. Contribuicao para a Fase 3", styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Os arquivos em llm_responses/*.json contem pares (prompt, resposta) com score de "
        "qualidade automatico, constituindo o dataset inicial para fine-tuning de um LLM "
        "especializado em saude feminina na Fase 3. O contexto medico feminino estabelecido "
        "nos prompts desta Fase 2 sera diretamente reutilizado como base para o assistente "
        "medico completo.",
        styles['body']
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── SECTION 11 ─────────────────────────────────────────────────────────
    story.append(section_header("11. Conclusao", styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "O Algoritmo Genetico demonstrou eficacia na otimizacao de hiperparametros do modelo "
        "de diagnostico de cancer de mama, superando o modelo original em todas as metricas "
        "relevantes. A melhoria de +2.38% no recall tem impacto clinico direto: aproximadamente "
        "24 mulheres diagnosticadas a mais por mes em escala hospitalar.",
        styles['body']
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "A integracao com LLM local (flan-t5-base) resolve o problema de comunicabilidade dos "
        "resultados, tornando o sistema utilizavel por profissionais de saude sem background "
        "tecnico em ML, com total privacidade dos dados da paciente.",
        styles['body']
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "O projeto atende integralmente aos requisitos do Tech Challenge Fase 2 - Projeto 1, "
        "incluindo: codificacao de genes, operadores geneticos, funcao fitness clinica com "
        "equidade, 3 experimentos comparativos, integracao LLM sensivel ao genero, avaliacao "
        "de qualidade das respostas e preparacao do dataset para a Fase 3.",
        styles['body']
    ))
    story.append(Spacer(1, 0.8 * cm))

    # Final red accent line
    final_bar = Table([['']], colWidths=[PAGE_W - 2 * MARGIN], rowHeights=[0.15 * cm])
    final_bar.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), RED)]))
    story.append(final_bar)
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "FIAP Pos Tech  -  Tech Challenge Fase 2  -  Marco 2026",
        ParagraphStyle('footer_note', fontName='Helvetica', fontSize=8,
                       textColor=HexColor('#90a4ae'), alignment=TA_CENTER)
    ))

    # ── Build ──────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=on_first, onLaterPages=on_later)
    print(f"PDF gerado com sucesso: {OUTPUT_PDF}")
    size_kb = os.path.getsize(OUTPUT_PDF) / 1024
    print(f"Tamanho do arquivo: {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    build_document()
