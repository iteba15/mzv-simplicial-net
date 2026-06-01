import json
from pathlib import Path


class ReportGenerator:
    """
    Reads hit_registry.json and renders a LaTeX preprint.
    Compile the output with: pdflatex output/zeta_hunter_report.tex
    """

    OUTPUT_DIR = Path("output")

    def __init__(self, registry_path: str = "hit_registry.json"):
        self.registry_path = Path(registry_path)

    def generate(self, output_path: str = "output/zeta_hunter_report.tex") -> str:
        """Render the LaTeX report. Returns the path written."""
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        hits = self._load_registry()
        sections = [
            self._preamble(),
            self._section_intro(),
            self._section_methodology(hits),
            self._section_results(hits),
            self._section_null_results(),
            self._section_conclusion(),
            self._appendix(hits),
            r"\end{document}",
        ]
        Path(output_path).write_text("\n".join(sections), encoding="utf-8")
        return output_path

    def _load_registry(self) -> list:
        if not self.registry_path.exists():
            return []
        with open(self.registry_path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _latex_escape(value) -> str:
        """Escape LaTeX special characters in user-supplied strings."""
        s = str(value)
        for char, repl in [
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ]:
            s = s.replace(char, repl)
        return s

    def _preamble(self) -> str:
        return (
            r"\documentclass[11pt,a4paper]{article}" + "\n"
            r"\usepackage{amsmath,amssymb,amsthm,booktabs,hyperref,longtable,geometry}" + "\n"
            r"\geometry{margin=2.5cm}" + "\n"
            r"\title{Computational Search for Closed Forms of Odd Riemann Zeta Values\\" + "\n"
            r"       \large Zeta Hunter --- Research Report}" + "\n"
            r"\author{Alvin Kigondu \\ Kigs Apex LLP}" + "\n"
            r"\date{\today}" + "\n"
            r"\begin{document}" + "\n"
            r"\maketitle" + "\n"
            r"\begin{abstract}" + "\n"
            "We report a systematic computational search for polynomial continued fraction (PCF)\n"
            r"representations of the odd Riemann zeta values $\zeta(3)$, $\zeta(5)$, $\zeta(7)$," + "\n"
            r"Catalan's constant $G$, and $4/\pi$. Three theory-guided families --- Ap\'{e}ry," + "\n"
            "Zagier, and Ramanujan --- were swept using a batched GPU evaluator (RTX 5080, float64)\n"
            "with mpmath verification at 500-digit precision. All hits above 50 digits are reported;\n"
            r"null results are documented as citable contributions." + "\n"
            r"\end{abstract}"
        )

    def _section_intro(self) -> str:
        return (
            r"\section{Introduction}" + "\n"
            r"The even Riemann zeta values satisfy the Euler formula $\zeta(2k) = (-1)^{k+1}"
            r"\frac{B_{2k}(2\pi)^{2k}}{2(2k)!}$, " + "\n"
            r"giving closed forms in terms of $\pi$. The odd values $\zeta(3), \zeta(5), \ldots$"
            " remain mysterious.\n"
            r"Ap\'{e}ry proved $\zeta(3) \notin \mathbb{Q}$ in 1978 via a polynomial continued"
            " fraction (PCF),\n"
            "but no elementary closed form is known for any odd zeta value.\n\n"
            "This report documents a GPU-accelerated search through theory-guided PCF families\n"
            r"near Ap\'{e}ry's construction and related Zagier and Ramanujan families."
        )

    def _section_methodology(self, hits: list) -> str:
        families = sorted(set(h.get("family", "Unknown") for h in hits)) or [
            "Apery", "Zagier", "Ramanujan"
        ]
        fstr = ", ".join(families)
        return (
            r"\section{Methodology}" + "\n"
            r"\subsection{Polynomial Continued Fractions}" + "\n"
            r"A PCF has the form $\text{CF}(a,b) = a(1) / (b(1) + a(2) / (b(2) + \cdots))$"
            " where $a(n)$ and $b(n)$ are polynomials with integer coefficients.\n\n"
            r"\subsection{Search Families}" + "\n"
            f"Three families were swept: {fstr}. "
            r"The Ap\'{e}ry family (degree 6/3) matches Ap\'{e}ry's original construction. "
            "The Zagier family (degree 2/2) is known to produce closed forms for "
            r"$\zeta(2)$, $\beta(3)$, and Catalan's constant at small coefficients. "
            "The Ramanujan family (degree 1/2) matches the shape of Ramanujan's $4/\\pi$ formula.\n\n"
            r"\subsection{Precision Pipeline}" + "\n"
            r"\textbf{Stage~1} (GPU float64): batches of 500{,}000 PCFs, depth 500,"
            r" threshold $10^{-8}$. "
            r"\textbf{Stage~2} (mpmath 500 digits): re-verify all Stage~1 hits; "
            r"classify by precision and coefficient size. "
            r"\textbf{Stage~3} (mpmath 2000 digits): manual confirmation of CANDIDATE hits"
            " with PSLQ follow-up."
        )

    def _section_results(self, hits: list) -> str:
        candidates = [
            h for h in hits
            if h.get("verdict") in ("CANDIDATE", "IDENTITY")
        ]
        if not candidates:
            return (
                r"\section{Results}" + "\n"
                "No hits above the 50-digit Stage~2 threshold were found in the completed sweeps.\n"
                r"See Section~\ref{sec:null} for the null-result summary."
            )
        rows = []
        for h in sorted(candidates, key=lambda x: -x.get("stage2_precision_digits", 0)):
            fam = self._latex_escape(h.get("family", "?"))
            tgt = self._latex_escape(h.get("target", "?"))
            prec = h.get("stage2_precision_digits", 0)
            a = self._latex_escape(str(h.get("a_coeffs", [])))
            b = self._latex_escape(str(h.get("b_coeffs", [])))
            verdict = self._latex_escape(h.get("verdict", "?"))
            rows.append(
                f"    {fam} & ${tgt}$ & {prec:.1f} & "
                f"\\texttt{{{a}}} & \\texttt{{{b}}} & {verdict} \\\\"
            )
        body = "\n".join(rows)
        return (
            r"\section{Results}" + "\n"
            r"Table~\ref{tab:hits} lists all Stage~2 verified hits." + "\n\n"
            r"\begin{table}[h]" + "\n"
            r"\centering" + "\n"
            r"\caption{Stage~2 verified PCF hits ($\geq 50$ digit precision).}" + "\n"
            r"\label{tab:hits}" + "\n"
            r"\begin{tabular}{llrllr}" + "\n"
            r"\toprule" + "\n"
            r"Family & Target & Digits & $a(n)$ coefficients & $b(n)$ coefficients & Verdict \\" + "\n"
            r"\midrule" + "\n"
            + body + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\end{table}"
        )

    def _section_null_results(self) -> str:
        return (
            r"\section{Null Results}" + "\n"
            r"\label{sec:null}" + "\n"
            "The absence of a hit within a searched region is a citable contribution.\n"
            "For each completed sweep, the run log records total combinations scanned,\n"
            r"coefficient range, depth, and elapsed time. These null results constrain"
            " any putative closed form:\n"
            r"if a PCF of the Ap\'{e}ry shape with coefficients in $[-60, 60]$ and"
            r" depth 500 matched $\zeta(3)$ to 8 digits, we would have found it."
        )

    def _section_conclusion(self) -> str:
        return (
            r"\section{Conclusion}" + "\n"
            r"This computational search confirms the mathematical consensus: no elementary"
            r" PCF closed form for $\zeta(3)$ was found within the searched families"
            " and coefficient ranges.\n"
            r"Future directions include extending the Ap\'{e}ry family to degree (8,4),"
            r" a multi-target lattice search linking $\zeta(3)$, $\zeta(5)$, $\zeta(7)$"
            " simultaneously (Broadhurst conjecture), and pursuing the Zagier family"
            r" at coefficient ranges $\pm 200$."
        )

    def _appendix(self, hits: list) -> str:
        if not hits:
            return r"\appendix" + "\n" + r"\section{Hit Registry}" + "\nNo hits recorded.\n"
        rows = []
        for h in hits:
            ts = self._latex_escape(h.get("timestamp", "")[:10])
            fam = self._latex_escape(h.get("family", "?"))
            tgt = self._latex_escape(h.get("target", "?"))
            prec = h.get("stage2_precision_digits", 0)
            verdict = self._latex_escape(h.get("verdict", "?"))
            rows.append(
                f"    {ts} & {fam} & ${tgt}$ & {prec:.1f} & {verdict} \\\\"
            )
        body = "\n".join(rows)
        return (
            r"\appendix" + "\n"
            r"\section{Full Hit Registry}" + "\n"
            r"\begin{longtable}{lllrl}" + "\n"
            r"\toprule" + "\n"
            r"Date & Family & Target & Digits & Verdict \\" + "\n"
            r"\midrule" + "\n"
            r"\endhead" + "\n"
            + body + "\n"
            r"\bottomrule" + "\n"
            r"\end{longtable}"
        )
