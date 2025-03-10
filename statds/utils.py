import inspect
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from . import no_parametrics, parametrics

try:
    from fpdf import FPDF
    from fpdf.fonts import FontFace
    available_fpdf = True
except ImportError:
    print("Warning - Don't available fpdf")
    available_fpdf = False
    FPDF, FontFace = None, None

current_directory = Path(__file__).resolve().parent


class LibraryError(Exception):
    def __init__(self, message):
        super().__init__(message)


properties = {"title-header": "Inform elaborated with StaTDS",
              "subtitle-header": "Version 1.1.5",
              "ref-library": "https://github.com/kdis-lab/StaTDS"}

available_test_multiple_groups = {"Friedman": no_parametrics.friedman,
                                  "Friedman + Iman Davenport": no_parametrics.iman_davenport,
                                  "Friedman Aligned Ranks": no_parametrics.friedman_aligned_ranks,
                                  "Quade": no_parametrics.quade,
                                  "Kruskal-Wallis": no_parametrics.kruskal_wallis,
                                  "ANOVA between cases": parametrics.anova_cases,
                                  "ANOVA within cases": parametrics.anova_within_cases
                                  }
available_test_two_groups = {"Wilcoxon": no_parametrics.wilcoxon,
                             "Binomial Sign": no_parametrics.binomial,
                             "Mann-Whitney U": no_parametrics.binomial,
                             "T-Test paired": parametrics.t_test_paired,
                             "T-Test unpaired": parametrics.t_test_unpaired
                             }

available_post_hoc = { "Nemenyi": no_parametrics.nemenyi,
                       "Bonferroni": no_parametrics.bonferroni,
                       "Li": no_parametrics.li,
                       "Holm": no_parametrics.holm,
                       "Holland": no_parametrics.holland,
                       "Finner": no_parametrics.finner,
                       "Hochberg": no_parametrics.hochberg,
                       "Hommel": no_parametrics.hommel,
                       "Rom": no_parametrics.rom,
                       "Schaffer": no_parametrics.shaffer
                       }

parametrics_test = ["T-Test paired", "T-Test unpaired", "ANOVA between cases", "ANOVA within cases"]

if available_fpdf:

    class InformPDF(FPDF):
        """ A custom class for creating PDF documents with specific formatting and layout features. """
        def __init__(self):
            """
            Initializes the dimensions of the FPDF document.

            Attributes
            ----------
            default_margin : float
                The default left margin of the document.
            margin_bottom_threshold : float
                The bottom margin threshold of the document, set to 10% of the document's height.
            """
            super().__init__()
            self.default_margin = self.l_margin
            self.margin_bottom_threshold = self.h * 0.10

        def check_page_break(self, h):
            """
            Checks if a page break is necessary before adding content with a specified height.

            Parameters
            ----------
            h : float
                Height of the element to be checked.

            Returns
            -------
            bool
                True if a new page needs to be added, otherwise False.
            """
            if self.get_y() + h > self.page_break_trigger:
                self.add_page()
                return True
            return False

        def header(self):
            """
            Defines the document header. The header includes an image on the left and text in three different rows on
            the right, configured with different styles and alignments.
            """

            # self.reset_margin()
            # Logo (imagen) a la izquierda
            self.image(current_directory / 'assets/images/logo-kdislab.png', 40, 15, 33)
            self.image(current_directory / 'assets/images/logo-StaTDS-without-background.png', 10, 4, 30)
            # Texto a la derecha en tres filas distintas
            self.set_font("helvetica", "B", 15)
            # Moving cursor to the right:
            self.set_xy(160, 10)
            self.cell(30, 10, properties["title-header"], align="R")
            # Performing a line break:
            self.set_xy(160, 15)
            self.set_font("helvetica", "I", 11)
            self.cell(30, 10, properties["subtitle-header"], align="R")
            self.set_xy(160, 20)
            self.set_font("helvetica", "I", 15)
            self.cell(30, 10, properties["ref-library"], align="R")
            self.ln(12)

        def footer(self):
            """
            Defines the document footer. The footer displays the current page number and the total number of pages.
            """ 
            self.set_y(-15)
            # Setting font: helvetica italic 8
            self.set_font("helvetica", "I", 8)
            self.set_text_color(128)
            # Printing page number:
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

        def print_header(self, text, font_size=16, font_style='B', fill=False, border="", align="L"):
            """
            Prints custom headers in the document based on the provided arguments.

            Parameters
            ----------
            text : str
                Text to be printed in the header.
            font_size : int, optional
                Font size for the header, default is 16.
            font_style : str, optional
                Font style for the header, default is 'B' (bold).
            fill : bool, optional
                Whether to fill the header background, default is False.
            border : str, optional
                Border style for the header, default is empty (no border).
            align : str, optional
                Text alignment for the header, default is 'L' (left-aligned).
            """
            self.set_font('Times', font_style, font_size)
            if fill:
                self.set_fill_color(200, 220, 255)
            self.cell(w=0, h=10, txt=text, new_x="LMARGIN", new_y="NEXT", border=border, fill=fill, align=align)
            self.set_fill_color(255, 255, 255)

        def reset_margin(self):
            """
            Resets the document's left margin to the default value.
            """
            self.set_left_margin(self.default_margin)


    def create_inform(pdf: InformPDF, title_experiment, title, test_subtitle, test_result, h_post_hoc, show_table,
                      show_graph, fig_size):
        """
        Creates an informative PDF document with specified content.

        Parameters
        ----------
        pdf : InformPDF
            An instance of InformPDF, a custom PDF generation class.
        title_experiment : str
            Title for the experiment section.
        title : str
            Title for the document.
        test_subtitle : str
            Subtitle for the test section.
        test_result : list
            List of test results to be displayed.
        h_post_hoc : str
            Post-hoc information or analysis to be included.
        show_table : pandas.DataFrame
            A DataFrame containing the test results.
        show_graph : str
            Path to an image file to be displayed.
        fig_size : tuple
            Size of the figure (width, height).
        """
        def create_table(show_table):
            data = [show_table.columns.tolist()] + show_table.values.tolist()

            # Establecer el ancho de las columnas
            col_width = pdf.w / (len(show_table.columns) * 2)
            # size_cols = [25, 15, 35, 15, 20, 35]
            size_cols = [25] * len(show_table.columns)
            with pdf.table(width=sum(size_cols), col_widths=size_cols, borders_layout="MINIMAL",
                           headings_style=FontFace(emphasis="B")) as t:

                aligns = ['C'] * len(show_table.columns)
                table_width = col_width * len(show_table.columns)
                x = (pdf.w - table_width) / 2

                # Establecer la posición x para la primera celda en la fila
                pdf.set_x(x)
                # Agregar filas a la tabla
                pdf.set_font("Times", "", 12)
                for i, row in enumerate(data):
                    row_table = t.row()
                    for item, align in zip(row, aligns):
                        text = str(round(item, 4)) if type(item) is float else str(item)
                        row_table.cell(text, align)

                    pdf.set_x(x)

        pdf.print_header(text=title_experiment, font_size=16, font_style='B', fill=False, align="C")
        pdf.print_header(text=title, font_size=14, font_style='B', fill=True, border="L")
        pdf.print_header(text=test_subtitle, font_size=12, font_style='BI')
        pdf.set_font("Times", "", 11)
        pdf.set_left_margin(pdf.l_margin * 1.5)
        tables_to_show = []
        for item in test_result:
            if type(item) is pd.DataFrame:
                tables_to_show.append(item)
                continue
            pdf.cell(txt=item, markdown=True)
            pdf.ln()

        # Show new tables
        for i in tables_to_show:
            create_table(i)

        pdf.reset_margin()

        if h_post_hoc != "":
            pdf.set_left_margin(pdf.l_margin * 2)
            pdf.print_header(text=h_post_hoc, font_size=12, font_style='B', fill=False)
            pdf.reset_margin()
            pdf.set_font("Times", "", 16)

            if not (show_table is None):
                create_table(show_table)

            if show_graph != "":
                adjustment_factor = [12, 18]
                x = (pdf.w - fig_size[0] * adjustment_factor[0]) / 2
                pdf.set_x(x)
                pdf.image(show_graph, h=fig_size[1] * adjustment_factor[1], w=fig_size[0] * adjustment_factor[0])
                pdf.ln()


def gets_args_functions(name_function):
    """
    Retrieves the arguments and their default values for a specified function.

    Parameters
    ----------
    name_function : function
        The function for which argument information is required.

    Returns
    -------
    dict
        A dictionary containing argument names as keys and their default values as values.
    """
    args_functions = inspect.signature(name_function)

    args = {name: parameter.default for name, parameter in args_functions.parameters.items()
            if name not in ["self", "kwargs", "args"]}

    return args


def print_results(title_experiment, title, test_subtitle, test_result, h_post_hoc, show_table, show_graph):
    """
    Prints the results and related information to the console.

    Parameters
    ----------
    title_experiment : str
        Title for the experiment section.
    title : str
        Title for the document.
    test_subtitle : str
        Subtitle for the test section.
    test_result : list
        List of test results to be printed.
    h_post_hoc : str
        Post-hoc information or analysis to be included.
    show_table : pandas.DataFrame
        A DataFrame containing the test results.
    show_graph : str
        Name of the saved graph or None if no graph is available.
    """
    print(title_experiment)
    print(title)
    print(test_subtitle)
    for item in test_result:
        show = item if type(item) is pd.DataFrame else f"\t {item}"
        print(show)

    if h_post_hoc != "":
        print(h_post_hoc)
        if not (show_table is None):
            print(show_table)

        if show_graph != "":
            print("Saved Graph with name")


def analysis_of_experiments(dataset, experiments: dict, generate_pdf: bool = False, name_pdf: str = "inform.pdf"):
    """
    Performs statistical analysis on experiments and optionally generates a PDF report.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset containing the experimental data.
    experiments : dict
        A dictionary defining the experiments and their parameters.
    generate_pdf : bool, optional
        Whether to generate a PDF report (default is False).
    name_pdf : str, optional
        Name of the PDF file (default is 'inform.pdf').
    """
    global available_test_multiple_groups, available_test_two_groups, available_post_hoc, parametrics_test
    pdf = None
    if generate_pdf and available_fpdf:
        pdf = InformPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=pdf.margin_bottom_threshold)
        pdf.set_title("Reports of statistics analysis")
        pdf.set_author("Library StaTDS")

    if type(experiments) is not dict:
        raise LibraryError("Error: Data structure not compatible with the expected format. Dictionary is expected. "
                           "It is recommended to follow the examples or use function generate_json")

    for name_experiment in experiments.keys():
        info_experiment = experiments[name_experiment]
        if type(info_experiment) is not dict:
            raise LibraryError("Error: Data structure not compatible with the expected format. Dictionary is expected.")

        multigroup_test = False

        type_test = "parametric" if info_experiment["test"] in parametrics_test else "no parametrics"

        if info_experiment["test"] in available_test_two_groups.keys():
            test_function = available_test_two_groups[info_experiment["test"]]
        elif info_experiment["test"] in available_test_multiple_groups.keys():
            test_function = available_test_multiple_groups[info_experiment["test"]]
            multigroup_test = True
        else:
            raise LibraryError("Error: Test no available")

        args = gets_args_functions(test_function)

        columns_to_analysis = list(dataset.columns)
        title = "Multiple Groups" if multigroup_test else "Two Groups"

        if multigroup_test is False:
            user_columns = [info_experiment["first_group"], info_experiment["second_group"]]
            missing_columns = set(user_columns) - set(columns_to_analysis)
            if len(missing_columns) != 0:
                raise LibraryError(f"Error: Missing columns in datasets -> {list(missing_columns)}")
            columns_to_analysis = user_columns

        if "alpha" not in info_experiment.keys():
            raise LibraryError("Error: alpha argument dont found")

        if multigroup_test and "minimize" not in info_experiment.keys():
            raise LibraryError("Error: criterion argument dont found")

        criterion = False if "minimize" not in info_experiment.keys() else info_experiment["minimize"]

        function_args = {"dataset": dataset[columns_to_analysis], "alpha": float(info_experiment["alpha"]),
                         "minimize": criterion, "verbose": False, "apply_correction": True}

        function_args = {i: function_args[i] for i in args.keys()}

        test_subtitle = f"{info_experiment['test']} test (significance level of {float(info_experiment['alpha'])})"
        post_hoc_test = False if "post_hoc" not in info_experiment.keys() or info_experiment[
            "post_hoc"] is None else True
        show_graph = ""
        show_table = None
        h_post_hoc = ""
        fig_size = []

        if multigroup_test:
            if info_experiment["test"] == "Kruskal-Wallis":
                statistic, p_value, critical_value, hypothesis = test_function(**function_args)
                table_results = None
            else:
                table_results, statistic, p_value, critical_value, hypothesis = test_function(**function_args)
            test_result = [f"- **Statistic:** {statistic}", f"- **Result:** {hypothesis}"]
            if p_value is None:
                test_result.extend([f"- **Critival Value:** {critical_value}"])
            else:
                test_result.extend([f"- **P-value:** {p_value}"])
            if type(table_results) is list:
                test_result.extend(table_results)
            elif type(table_results) is dict:
                rankings_data = {i[0]: [round(i[1], 5)] for i in table_results.items()}
                test_result.append(pd.DataFrame(rankings_data))
            if post_hoc_test:
                if info_experiment["post_hoc"] not in available_post_hoc.keys():
                    raise LibraryError("Error: Post Hoc no available")

                post_hoc_test = available_post_hoc[info_experiment["post_hoc"]]

                args = gets_args_functions(post_hoc_test)

                control = info_experiment["control"] if "control" in info_experiment.keys() else None
                all_vs_all = info_experiment["all_vs_all"] if "all_vs_all" in info_experiment.keys() else True

                function_args = {"ranks": table_results, "num_cases": dataset.shape[0],
                                 "alpha": float(info_experiment["alpha"]), "control": control,
                                 "all_vs_all": all_vs_all, "verbose": False,
                                 "type_rank": info_experiment["test"]
                                 }

                function_args = {i: function_args[i] for i in args.keys()}

                h_post_hoc = (f"Post hoc: {info_experiment['post_hoc']} test (significance level of "
                              f"{float(info_experiment['alpha'])})")
                results = post_hoc_test(**function_args)

                if type(results) is pd.DataFrame:
                    show_table = results
                else:
                    if type(results[0]) is pd.DataFrame:
                        show_table = results[0]
                    fig = results[-1]
                    buf = io.BytesIO()  # in-memory files
                    fig_size = fig.get_size_inches()
                    # TODO CAMBIAR EL COMO SE GUARDAN LOS GRÁFICOS CUANDO NO SE GENERA EL INFORME
                    fig.savefig(buf, format="png")
                    data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
                    buf.close()
                    show_graph = "data:image/png;base64,{}".format(data)
        else:
            statistic, critical_value, p_value, hypothesis = test_function(**function_args)
            test_result = [f"- **Statistic:** {statistic}", f"- **Result:** {hypothesis}"]
            if p_value is None:
                test_result.extend([f"- **Critival Value:** {critical_value}"])
            else:
                test_result.extend([f"- **P-value:** {p_value}"])

        if generate_pdf and available_fpdf:
            create_inform(pdf, f"Experiment {name_experiment} : {type_test}", title, test_subtitle, test_result,
                          h_post_hoc, show_table, show_graph, fig_size)
        else:
            print_results(f"Experiment {name_experiment} : {type_test}", title, test_subtitle, test_result,
                          h_post_hoc, show_table, show_graph)

        plt.close('all')

    if generate_pdf and available_fpdf:
        pdf.output(name_pdf)


def process_alpha_experiment(parameter):
    """
    Processes and validates experiment parameters, including alpha level and test details.

    Parameters
    ----------
    parameter : dict
        A dictionary containing experiment parameters.

    Returns
    -------
    dict
        A dictionary containing processed and validated experiment parameters.

    Raises
    ------
    LibraryError
        If the provided parameters are invalid or missing.
    """
    available_alpha = [0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
    name_test_two_groups = ["Wilcoxon", "Binomial Sign", "Mann-Whitney U", "T-Test paired", "T-Test unpaired"]
    name_test_multiple_groups = ["Friedman", "Friedman Aligned Ranks", "Quade", "ANOVA between cases",
                                 "ANOVA within cases", "Kruskal-Wallis", "Friedman + Iman Davenport"]

    struct = {}

    if not (float(parameter["alpha"]) in available_alpha):
        raise LibraryError(f"Error: {parameter['alpha']} is not found in the list of available alphas")

    struct["alpha"] = float(parameter["alpha"])

    if not ("test" in parameter.keys()):
        raise LibraryError(f"Error: Missing argument minimize for the required test")

    struct["test"] = parameter["test"]

    if struct["test"] in name_test_two_groups:
        if not ("first_group" in parameter.keys()):
            raise LibraryError(f"Error: Missing argument first_group for the required test")
        elif not ("second_group" in parameter.keys()):
            raise LibraryError(f"Error: Missing argument second_group for the required test")

        struct["first_group"] = parameter["first_group"]
        struct["second_group"] = parameter["second_group"]

    elif struct["test"] in name_test_multiple_groups:
        if not ("minimize" in parameter.keys()):
            raise LibraryError(f"Error: Missing argument minimize for the required test")
        struct["minimize"] = parameter["minimize"]
        struct["post_hoc"] = parameter["post_hoc"] if "post_hoc" in parameter.keys() else None
        struct["control"] = parameter["control"] if "control" in parameter.keys() else None

    return struct


def generate_json(names_experiments: list, parameters_experiments: list):
    """
    Generates a JSON structure containing experiment parameters.

    Parameters
    ----------
    names_experiments : list
        List of experiment names.
    parameters_experiments : list
        List of dictionaries containing experiment parameters.

    Returns
    -------
    dict
        A dictionary representing experiment parameters in a structured JSON format.
    """
    struct_json = {}
    for index, value in enumerate(zip(names_experiments, parameters_experiments)):
        experiment, parameter = value
        if type(parameter["alpha"]) is str or type(parameter["alpha"]) is float:
            parameter["alpha"] = float(parameter["alpha"])
            struct_json[experiment] = process_alpha_experiment(parameter)
        elif type(parameter["alpha"]) is list:
            list_of_alpha = parameter["alpha"]
            for alpha in list_of_alpha:
                parameter["alpha"] = float(alpha)
                struct_json[experiment + f" with alpha {alpha}"] = process_alpha_experiment(parameter)
    return struct_json


def dataframe_to_latex(dataframe: pd.DataFrame, caption: str = None, label: str = None):
    """
    Converts a pandas DataFrame into LaTeX code for table creation.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The pandas DataFrame to be converted.
    caption : str, optional
        The table caption in LaTeX.
    label : str, optional
        The table label in LaTeX.

    Returns
    -------
    str
        LaTeX code for creating a table.
    """
    latex_code = "\\begin{table}[ht]\n\\centering\n"

    if caption:
        latex_code += f"\\caption{{{caption}}}\n"

    if label:
        latex_code += f"\\label{{{label}}}\n"

    latex_code += "\\begin{tabular}{|"

    column_formats = {'int64': 'c', 'float64': 'c', 'object': 'l'}

    for column in dataframe.columns:
        column_type = dataframe[column].dtype
        latex_code += f"{column_formats.get(str(column_type), 'l')}|"

    latex_code += "}\n\\hline\n"

    latex_code += " & ".join(dataframe.columns) + " \\\\ \\hline\n"

    for _, row in dataframe.iterrows():
        latex_code += " & ".join([str(value) for value in row]) + " \\\\\n"

    latex_code += "\\end{tabular}\n\\end{table}"

    return latex_code


if __name__ == "__main__":
    analysis_form = {
        "experiment_1": {
            "test": "Wilcoxon",
            "alpha": 0.05,
            "first_group": "PDFC",
            "second_group": "FH-GBML",
            "minimize": False
        },
        "experiment_4": {
            "test": "Friedman",
            "alpha": 0.05,
            "post_hoc": "Holm",
            "minimize": False,
            "control": None,
            "all_vs_all": True
        },
        "experiment_2": {
            "test": "Friedman",
            "alpha": 0.05,
            "post_hoc": "Nemenyi",
            "minimize": False,
            "control": None,
            "all_vs_all": False
        },
        "experiment_3": {
            "test": "Friedman",
            "alpha": 0.05,
            "post_hoc": None,
            "minimize": False,
            "control": None,
            "all_vs_all": True
        },
        "experiment_5": {
            "test": "T-Test paired",
            "alpha": 0.05,
            "first_group": "PDFC",
            "second_group": "FH-GBML",
            "minimize": False
        }
    }
    import pandas as pd
    df = pd.read_csv("assets/app/sample_dataset.csv")
    # analysis_of_experiments(df, analysis_form, generate_pdf=True)
    columns = list(df.columns)

    aux = generate_json(["A", "B", "C", "D", "E"],
                        [{"alpha": 0.05, "test": "Wilcoxon", "first_group": columns[1],
                          "second_group": columns[-1]},
                         {"alpha": "0.05", "test": "T-Test paired",
                          "first_group": columns[1], "second_group": columns[-1]},
                         {"alpha": 0.05, "test": "ANOVA between cases", "minimize": True},
                         {"alpha": [0.05, 0.01], "test": "Friedman", "minimize": True, "post_hoc": "Bonferroni"},
                         {"alpha": [0.05, 0.01], "test": "Friedman", "minimize": True, "post_hoc": "Nemenyi"}
                         ])
    analysis_of_experiments(df, aux, generate_pdf=True)
