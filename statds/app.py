try:
    import dash
    from dash import Dash, html, dcc, Input, Output, State
    import dash_bootstrap_components as dbc
    import pandas as pd
    import dash_daq as daq
    import dash_ag_grid as dag
    import inspect
    import io
    import base64
    from datetime import datetime
    from flask import Flask
    available_app = True
except ImportError:
    print("Warning - Don't available dash, so you can't use web service. "
          "If you need use web service must install statds with pip statds[full-app]")
    available_app = False

if available_app:
    import multiprocessing
    from pathlib import Path
    import os

    from . import no_parametrics, parametrics, utils
    from . import normality, homoscedasticity

    current_directory = Path(__file__).resolve().parent
    external_stylesheets = [dbc.themes.BOOTSTRAP, "app/style.css",
                            'https://use.fontawesome.com/releases/v5.8.1/css/all.css']

    server = Flask(__name__)
    server.secret_key = 'test'
    my_root = os.getenv("WSGI_APPURL", "") + "/"
    my_root = os.getenv("DASH_REQUESTS_PATHNAME_PREFIX", my_root)
    # assets_path = os.getcwd() +'/assets'
    # app = Dash(__name__, server = server, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True,
    # url_base_pathname=my_root)

    app = Dash(__name__, server=server, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.title = 'StaTDS: Statistical Tests for Data Science'
    app._favicon = "images/logo-StaTDS.png"
    # suppress_callback_exceptions=True Esto no es una buena práctica pero es la única forma de mantener el control dinámico


    with open(current_directory / "assets/app/README.md", 'r') as film:
        readme_content = film.read()


    reference_content = ("Christian Luna Escudero, Antonio Rafael Moya Martín-Castaño, José María Luna Ariza, " +
                         "Sebastián Ventura Soto, StaTDS: Statistical Tests for Data Science (name article and journay)")

    bibtext_content = '''
                        ```latex
                        @InProceedings{statds, 
                            author={Christian Luna Escudero, Antonio Rafael Moya Martín-Castaño, José María Luna Ariza, Sebastián Ventura Soto},
                            title={StaTDS library: Statistical Tests for Data Science}, 
                            booktitle={Neurocomputing}, 
                            year={2024}
                        }
                        ```
                      '''


    def generate_text_sample():
        with open(current_directory / "assets/app/sample_dataset.csv", "r") as f:
            text = f.read()
        return text


    def generate_navigator_menu():
        menus_2 = [
                    {"title": html.Img(src="https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/logo-StaTDS-without-background.png", style={"width": "5em",
                                                                                                      "height": "5em"}),
                     "href": "home", "icon": "mdi:home-outline", "className": "logo-menu"},
                    {"title": html.Img(src="https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/logo-kdislab.png", style={"width": "4.25em", "height": "2em",
                                                                                    "margin-top": "1.55em",
                                                                                    "margin-left": "0.425em"
                                                                                    }),
                     "href": "https://www.uco.es/kdis/", "icon": "mdi:home-outline", "className": "logo-kdis-menu"},
                  ]
        menus = [
                    {"title": "Data Analysis", "href": "data_analysis", "className": "item-menu"},
                    {"title": "Normality & Homoscedasticity", "href": "normality_homoscedasticity",
                     "className": "item-menu"},
                ]
        import_button = dbc.Button(children="Import Data", id="import-Data", outline=True, className="menu item-menu")
        export_button = dbc.Button(children="Export Results", id="export-data", outline=True, className="menu item-menu")
        content = [dcc.Link(children=[html.Div(item["title"], className=item["className"])],
                            href=item["href"] if "href" in item.keys() else None, className="menu") for item in menus]

        content_2 = [dbc.Button(children=[html.Div(item["title"], className=item["className"])],
                                href=item["href"] if "href" in item.keys() else None, className="logos-menu", color=None)
                     for item in menus_2]

        download_file = dcc.Download(id="download-text")
        content = [html.Div(content_2, className="left-menu")] + content + [import_button, export_button,
                                                                            generate_import_data_page(),
                                                                            generate_export_table_page(), download_file]
        return html.Div(
            children=content, className="navigator_menu"
        )


    def generate_import_data_page():
        info_import_data = [
            dbc.PopoverHeader(html.H3("Data Import Help Guide")),
            dbc.PopoverBody([
                html.H5("Choosing the Data Separator"),
                dcc.Markdown('''
                        1. Identify the separator used in your dataset (e.g., comma, semicolon, tab).
                        2. Select the corresponding separator from the 'Type of separator' dropdown menu.
                        3. Available format is .csv, .txt and .xls.
                        ''', mathjax=True),
                html.H5("Option A: Drag and Drop:"),
                dcc.Markdown('''
                        1. Drag your file from your computer. 
                        2. Drop it into the designated 'Drag and Drop or Select File' area.
                        ''', mathjax=True),
                html.H5("Option B: Manual Entry"),
                dcc.Markdown('''
                        Click inside the 'Insert Dataset' text box.
                        Enter your data manually, ensuring that it matches the chosen separator format.
                        ''', mathjax=True)],
                style={"overflow-y": "scroll", "height": "30em"}
            ),
        ]
        example_of_dataset = generate_text_sample()

        modal = dbc.Modal(
                [
                    dbc.ModalHeader([dbc.ModalTitle("Import Dataset to analysis"),
                                     html.Button(html.I(className="fas fa-info-circle"), id="info_import_data", n_clicks=0,
                                                 className="button-info"),
                                     dbc.Popover(info_import_data, target="info_import_data", body=True, trigger="legacy")]),
                    dbc.ModalBody(["Type of separator:",
                                   dbc.Select(
                                       id="select-separator",
                                       options=[
                                           {"label": 'Comma (",")', "value": ","},
                                           {"label": 'Semicolon (";")', "value": ";"},
                                           {"label": 'Tab ("\\t")', "value": "\\t"},
                                           {"label": 'Space (" ")', "value": " "},
                                           {"label": 'Pipe ("|")', "value": "|"},
                                       ],
                                       value=","
                                   )
                                   ]),
                    dbc.ModalBody(["Insert Dataset",
                                   dbc.Textarea(id="textarea-dataset", size="lg", placeholder=example_of_dataset,
                                                style={'height': '200px'}, className="textarea")]),
                    dbc.ModalBody([
                                   dcc.Upload(
                                       id='upload-data',
                                       children=html.Div([
                                           'Drag and Drop or Select File'
                                       ]),
                                       style={
                                           'width': '95%',
                                           'height': '60px',
                                           'lineHeight': '60px',
                                           'borderWidth': '1px',
                                           'borderStyle': 'dashed',
                                           'borderRadius': '5px',
                                           'textAlign': 'center',
                                           'margin': '10px'
                                       },
                                       # Allow multiple files to be uploaded
                                       multiple=False
                                   )
                                   ]),
                    dbc.Button("Import Data", id="import-button")
                ],
                id="modal-xl",
                size="xl",
                is_open=False,
                scrollable=True,
            )
        main_content = html.Div(children=[modal])

        return main_content


    def generate_export_table_page():

        modal = dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Export Results")),
                    dbc.ModalBody("Here, the results of all the tables generated by the experiments will be displayed.",
                                  id="body-modal-export")
                ],
                id="modal-export",
                size="xl",
                is_open=False,
                scrollable=False,
            )
        main_content = html.Div(children=[modal])

        return main_content


    def generate_tabla_of_dataframe(df: pd.DataFrame, height_table: str = '30em'):
        if df is None or df.empty:
            return ""
        height_table = "7.5em" if df.shape[0] == 1 else "14em" if df.shape[0] < 5 else '30em'
        return dag.AgGrid(
            id="table",
            rowData=df.to_dict("records"),
            columnDefs=[
                {'headerName': col, 'field': col} for col in df.columns
            ],
            defaultColDef={"resizable": True, "filter": False, "minWidth": 150, "floatingFilter": False},
            columnSize="responsiveSizeToFit",
            style={'height': height_table, "margin-bottom": "1em"}
        )


    def generate_home_page(dataframe: pd.DataFrame):
        global readme_content, reference_content, bibtext_content

        info_authors = [
            {"title": "Christian Luna Escudero",
             "description": "gradua"
                            "ted in Computer Science with honors at the University of Córdoba (Spain) in 2023. He "
                            "is currently studying the Master's Degree in Research in Artificial Intelligence | AEPIA, "
                            "while working in the Knowledge Discovery and Intelligent Systems (KDIS) research group.",
             "email": "c.luna@uco.es", "image": "https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/i82luesc.png"},
            {"title": "Antonio Rafael Moya Martín-Castaño",
             "description": "is currently a Substitute Professor of Computing Sciences and Numerical Analysis at the University " 
                            "of Córdoba, while working in the Knowledge Discovery and Intelligent Systems (KDIS) research group. "
                            "His research specializes in hyper-parameter Optimization in Machine Learning models, focusing on deep "
                            "learning models in this task",
             "email": "amoya@uco.es", "image": "https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/amoya.png"},
            {"title": "José María Luna Ariza",
             "description": "is an Associate Professor at the Department of Computer Science and Numerical Analysis, University of Cordoba, "
                            "Spain. He received the Ph.D. degree in Computer Science from the University of Granada (Spain) in 2014. He is a "
                            "member of the the Knowledge Discovery and Intelligent Systems (KDIS) research group and the Andalusian Research Institute "
                            "in Data Science and Computational Intelligence (DaSCI). He is author of the two books related to pattern mining, published "
                            "by Springer. He has published more than 35 papers in top ranked journals and international scientific conferences, and he is"
                            " author of two book chapters. He has also been engaged in 4 national and regional research projects. Dr. Luna has contributed to "
                            "3 international projects. His research is focused on evolutionary computation and pattern mining.",
             "email": "jmluna@uco.es", "image": "https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/jmluna.png"},
            {"title": "Sebastián Ventura Soto",
             "description": "is currently a Full Professor in the Department of Computer Science and Numerical Analysis at the University of "
                            "Córdoba, where he heads the Knowledge Discovery and Intelligent Systems (KDIS) Research Laboratory. He is currently "
                            "the Deputy Director of the Andalusian Research Institute in Data Science and Computational Intelligence (DaSCI). "
                            "He received his B.Sc. and Ph.D. degrees in sciences from the University of Córdoba, Spain, in 1989 and 1996, "
                            "respectively. He has published three books and about 300 papers in journals and scientific conferences, and he has edited "
                            "three books and several special issues in international journals. He has also been engaged in 15 research projects (being the "
                            "coordinator of seven of them) supported by the Spanish and Andalusian governments and the European Union. His main research "
                            "interests are in the fields of data science, computational intelligence, and their applications. Dr. Ventura is a senior member of "
                            "the IEEE Computer, the IEEE Computational Intelligence and the IEEE Systems, Man and Cybernetics Societies, as well as the Association "
                            "of Computing Machinery (ACM).",
             "email": "sventura@uco.es", "image": "https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/sventura.png"},
        ]

        row_authors = [dbc.Row([
            dbc.Col(dbc.CardImg(src=author["image"], className="img-fluid rounded-start"), className="col-md-4"),
            dbc.Col(dbc.CardBody(
                [html.H4(author["title"], className="card-title"),
                 html.P(author["description"], className="card-text"),
                 html.Small(html.A(author["email"], href="mailto:" + author["email"]), className="card-text text-muted")]
            ), className="col-md-8"
            )
        ], className="g-0 d-flex align-items-center")
            for author in info_authors]
        about_authors = [dbc.Card(i, className="mb-3 card", style={"width": "40em", "margin-right": "1em"}) for i in
                         row_authors]
        foother_authors = html.Div(about_authors, className="container_card")
        main_content = html.Div(
            children=[
                html.H1(children='StaTDS: Statistical Tests for Data Science', className="title-app"),
                html.Div(id="table-content", children=generate_tabla_of_dataframe(dataframe),
                         className="table-info hidden" if dataframe is None or dataframe.empty else "table-info"),
                html.Div(children=[html.P(readme_content)],
                         className="content-info"),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Reference", className="card-title"),
                            html.P(reference_content, className="card-text"),
                            dcc.Markdown(bibtext_content)
                        ]
                    ),
                    className="content-info hidden",
                ),

                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button([html.Img(src='https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/logo-send-email.png', className="icon_button"),
                                    "Contact email"], href="mailto:c.luna@uco.es?cc=sventura@uco.es&subject=StaTDS",
                                   className="button", color="secondary", outline=True),
                        dbc.Button([html.Img(src='https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/logo-github.png', className="icon_button"),
                                    "Source on Github"], href="https://github.com/kdis-lab/StaTDS",
                                   className="button", color="secondary", outline=True),
                        dbc.Button([html.Img(src='https://raw.githubusercontent.com/kdis-lab/StaTDS/main/statds/assets/images/logo-python.png', className="icon_button"),
                                    "Python Doc"], href="https://statds.readthedocs.io/en/latest/", className="button",
                                   color="secondary", outline=True),
                        ]),
                ], className="p-4"),
                foother_authors
            ]
        )

        return main_content


    def home_page(dataframe: pd.DataFrame = None):
        return generate_home_page(dataframe)


    def analysis_page(dataframe: pd.DataFrame = None):
        return generate_home_page(dataframe)


    app.layout = html.Div(children=[
        dcc.Store(id='user-data', storage_type='session'),
        dcc.Store(id='user-file', storage_type='session'),
        dcc.Store(id='user-experiments', storage_type='session'),
        dcc.Location(id='url', refresh=False),
        generate_navigator_menu(),
        html.Div(id="page-content", children=home_page(), className="content_page")
    ])


    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open


    app.callback(
        Output("modal-xl", "is_open"),
        Input("import-Data", "n_clicks"),
        State("modal-xl", "is_open"),
    )(toggle_modal)


    app.callback(
        Output("modal-export", "is_open"),
        Input("export-data", "n_clicks"),
        State("modal-export", "is_open"),
    )(toggle_modal)


    def parse_contents(contents, filename, separator):
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        df = None
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')), sep=separator)
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'txt' in filename:
                # Assume that the user uploaded an txt file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')), sep=separator)
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

        return df


    def dataframe_to_text(df: pd.DataFrame, sep: str):
        column_names = df.columns.tolist()
        data_rows = [sep.join(map(str, row)) for _, row in df.iterrows()]

        text = "\n".join([sep.join(column_names)] + data_rows)

        return text


    @app.callback(Output('textarea-dataset', 'value'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('select-separator', 'value'))
    def update_output(list_of_contents, list_of_names, separator):
        if list_of_contents is not None:
            df = parse_contents(list_of_contents, list_of_names, separator)
            return dataframe_to_text(df, separator)

        return ""


    @app.callback(Output('textarea-dataset', 'placeholder'),
                  Input('select-separator', 'value'))
    def change_example(separator):
        placeholder = generate_text_sample()
        separator = "\t" if separator == "\\t" else separator
        placeholder = placeholder.replace(",", separator)

        return placeholder


    @app.callback(Output('import-button', 'n_clicks'),
                   Output('user-data', 'data'),
                  Output("import-Data", "n_clicks"),
                  Output('url', 'pathname'),
                  Input('import-button', 'n_clicks'),
                  State('textarea-dataset', 'value'),
                  State('select-separator', 'value'),
                  State("import-Data", "n_clicks"),
                  State('user-data', 'data'))
    def import_data(n_clicks, textarea_value, separator, n_clicks_modal, current_session):
        global my_root
        if n_clicks is None:
            return dash.no_update

        if n_clicks:
            dataframe = pd.read_csv(io.StringIO(textarea_value), sep=separator)
            return None, dataframe.to_dict(), n_clicks_modal + 1, my_root

        return None, current_session, n_clicks_modal, my_root


    def generate_alpha_form(multiple_alpha: bool = False, switch_selector: bool = True):
        available_alpha = [0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
        title = "Alpha(s)" if multiple_alpha else "Alpha"
        return [html.Div([html.Label(title, style={"margin-left": "1em", "margin-right": "1em"}),
                          dcc.Dropdown(
                            options=[{'label': str(i), 'value': str(i)} for i in available_alpha],
                            value=str(available_alpha[1]), style={'width': 'auto', "min-width": "8em", "max-width": "16em"},
                            id="selector_alpha", multi=multiple_alpha)]),
                html.Div([
                    html.Label('Optimization Criterion'),
                    html.Div([
                            html.Label('Min', style={"margin-right": "0.95em", "margin-left": "0.95em"}),
                            daq.BooleanSwitch(id='criterion_switch', on=False),
                            html.Label('Max', style={"margin-left": "1em"})
                        ], style={'display': 'flex', 'align-items': 'center'})
                ], className="" if switch_selector else "hidden")
                ]


    def generate_select_groups(columns: list, id_selector: str, text_label: str):
        available_groups = [{'label': i, 'value': i} for i in columns]
        return [
            html.Label(text_label),
            dcc.Dropdown(
                options=available_groups,
                style={"width": "25em"},
                id=id_selector
            ),
        ]


    def generate_selector_test(available_test: list, id_selector: str):
        return [
            html.Label("Test"),
            dcc.Dropdown(
                options=available_test,
                style={"width": "25em"},
                id=id_selector),
        ]


    def generate_post_hoc(available_test: list):
        available_test = [] if available_test is None else available_test
        return [
            html.Label("Post-hoc"),
            dcc.Dropdown(
                options=available_test,
                style={"width": "25em"},
                id="selector_post_hoc")
        ]


    def create_test_form(columns: list, title: str, test_two_groups: list, test_multiple_groups: list,
                         test_post_hoc: list = None, multi_alpha: bool = False):
        class_name_post_hoc = "hidden" if test_post_hoc is None else "form-element"

        info_pairwise_test = [
            dbc.PopoverHeader(html.H3("Pairwise Test")),
            dbc.PopoverBody([
                html.P("They allow determining if two algorithms have similar behaviour. Comparison of algorithms across "
                       "several data sets, which allows the analysis of whether the effectiveness of the algorithms varies "
                       "with the specific dataset and if one demonstrates consistently better "
                       "performance across different contexts."),
                html.H3("Parametrics Test"),
                html.H5("T-test"),
                dcc.Markdown('''The t-test, commonly referred to as Student's t-test, is tailor-made for assessing the disparities in average values between two groups. It examines the null hypothesis that two correlated or repeated samples share the same mean values. 
                    The test determines if there is a significant difference in average scores between the samples. It is applicable to: **Paired data:** the data values of each group are related. **Unpaired data:** the data values of each group are independent.            
                ''', mathjax=True),
                html.H3("Non-Parametrics Test"),
                html.H5("Wilcoxon"),
                dcc.Markdown('''The Wilcoxon signed-ranks test is a non-parametric alternative to the paired t-test, which ranks the differences in performances of
                            two samples for each data set, ignoring the signs and compares the ranks
                            for the positive and the negative differences''', mathjax=True),
                html.H5("Binomial Sign"),
                dcc.Markdown('''The Binomial Sign Test evaluates the null hypothesis asserting that two related paired samples 
                    originate from an identical distribution. Unlike other tests, it does not presuppose symmetry in the data. 
                    However, it is generally considered to be less robust compared to the Wilcoxon test''', mathjax=True),
                html.H5("Mann-Whitney-U"),
                dcc.Markdown('''In statistics, the Mann–Whitney U test (also known as the Mann–Whitney–Wilcoxon (MWW/MWU)) is a 
                    non-parametric test of the null hypothesis that for randomly selected values X and Y from two populations, 
                    the probability of X being greater than Y is equal to the probability of Y being greater than X. It is an 
                    alternative to the t-test for independent samples when the data do not meet the requirements of normality 
                    and homogeneity of variances required by the t-test. 
                    ''')
                ],
                                     style={"overflow-y": "scroll", "height": "30em"}  # Ajusta la altura según necesites
            ),
        ]
        info_multiple_test = [
            dbc.PopoverHeader(html.H3("Multiple Comparisons")),
            dbc.PopoverBody([
                html.P("It allows determining if various algorithms behave similarly across multiple data sets."),
                html.H3("Parametrics Test"),
                html.H5("ANOVA"),
                dcc.Markdown('''Anova examines the null hypothesis that the average outcomes of two or more groups are equivalent. The test scrutinizes both the variability among the groups 
                    and the internal variation within them using variance analysis. The statistical measure used in the ANOVA test is derived from the f-distribution.
                    '''),
                html.H3("Non-Parametrics Test"),
                html.H5("Friedman"),
                dcc.Markdown('''The Friedman Test is a non-parametric statistical method used for comparing and ranking multiple data sets. It calculates rankings for each set and evaluates the results using a chi-squared distribution. The degrees of freedom for this distribution are determined by $$K−1$$, where $$K$$ represents the number of related variables, such as the number of algorithms being compared. ''', mathjax=True),
                html.H5("Friedman + Iman Davenport"),
                dcc.Markdown('''The Friedman test, combined with the Iman-Davenport correction, offers a robust non-parametric alternative for analyzing repeated measures designs and comparing multiple algorithms or treatments. The Friedman test assesses the differences in rankings across related samples or matched groups. When the Friedman test indicates significant differences, the Iman-Davenport correction is applied to adjust the test statistic, providing a more accurate p-value. This correction accounts for the inherent conservatism of the Friedman test, making the results more reliable, especially with a small sample size or when dealing with datasets that violate the assumptions of parametric tests.''', mathjax=True),
                html.H5("Friedman Aligned Ranks"),
                dcc.Markdown('''Friedman's Aligned Ranks Test, an extension of the Friedman Test, also compares and assigns rankings across all data sets. This test is particularly useful when dealing with a smaller number of algorithms in the comparison. It provides a more comprehensive view by considering the collective performance of all sets.''', mathjax=True),
                html.H5("Quade"),
                dcc.Markdown('''The Quade Test, while similar in function to the Iman-Davenport Test, incorporates a unique aspect. It accounts for the varying difficulty levels of problems or for the more pronounced discrepancies in results obtained from different algorithms. This is achieved through a weighting process, which adds an extra layer of analysis, especially beneficial in scenarios where algorithms perform inconsistently across different types of problems.''', mathjax=True),
                html.H5("Kruskal-Wallis"),
                dcc.Markdown('''The Kruskal-Wallis Test is a non-parametric statistical method used for comparing medians across multiple independent groups. Unlike traditional ANOVA, it does not assume a normal distribution, making it suitable for ordinal or non-normally distributed data. The test calculates the ranks of all data points and evaluates whether the medians of the groups are significantly different, based on the sum of these ranks. It's particularly useful when dealing with three or more groups and provides an alternative to the one-way ANOVA when the data violates its assumptions.''', mathjax=True),
                html.H3("Post-hoc"),
                html.P("Once a Multiple Comparisons analysis has been performed, if significant differences arise after "
                       "concluding, post-hoc tests are necessary. Post-hoc tests determine where our differences come from,"
                       " and it is possible to consider a comparison among all pairs of algorithms or a comparison between "
                       "a control algorithm and the rest."),
                html.H5("Bonferroni-Dunn"),
                dcc.Markdown('''
                                The Bonferroni-Dunn test is a one-step method that adjusts the $$\\alpha$$ value based on the number of comparisons, using the formula $$\\alpha' = \\alpha/K$$, where $$K$$ represents the number of comparisons.
                                * For a comparison against a control: $$K = k-1$$
                                * For an all pairs comparison: $$K = k(k-1)/2$$
                            ''', mathjax=True),
                html.H5("Holm"),
                dcc.Markdown('''
                                The Holm test operates as a step-down procedure, systematically adjusting the value of $$\\alpha$$. Begin by ordering the p-values from smallest to largest: $$p_1 \\leq p_2 \\leq ... \\leq p_{k-1}$$. Correspondingly, assign these to hypotheses $$H_1, H_2, ..., H_{k-1}$$. In the Holm procedure:
                                * Initiate with the most significant p-value.
                                * Reject $H_1$ if $p_1$ is less than $\\alpha / (k-1)$. After which, evaluate $p_2$ against $\\alpha / (k-2)$.
                                * Continue this method sequentially. If, for instance, the second hypothesis is rejected based on its p-value, move to the third hypothesis.
                                * The process stops when we retain a null hypothesis. We also keep all subsequent without further testing.
                            ''', mathjax=True),
                html.H5("Holland"),
                dcc.Markdown('''
                                Holland test also adjust the value of $$\\alpha$$ in a step-down manner, as Holm'm method does. Begin by ordering the p-values from smallest to largest: $$p_1 \\leq p_2 \\leq ... \\leq p_{k-1}$$. Correspondingly, assign these to hypotheses $$H_1, H_2, ..., H_{k-1}$$. In the Holland procedure:
                                * Initiate with the most significant p-value.
                                * Reject $H_1$ if $p_1$ is less than $1 - (1-\\alpha)^{k-1}$. After which, evaluate $p_2$ against $1 - (1-\\alpha)^{k-2}$.
                                * Continue this method sequentially. If, for instance, the second hypothesis is rejected based on its p-value, move to the third hypothesis.
                                * The process stops when we retain a null hypothesis. We also keep all subsequent without further testing.
                            ''', mathjax=True),
                html.H5("Finner"),
                dcc.Markdown('''
                                Finner test is similar to Holm and also adjusts the value of $$\\alpha$$ in a step-down manner. In the Finner procedure:
                                * Initiate with the most significant p-value.
                                * Reject $$H_1$$ if $$p_1$$ is less than $$1 - (1-\\alpha)^{(k-1)/i}$$. After which, evaluate $$p_2$$ against $$\\alpha / (k-2)$$.
                                * Continue this method sequentially. If, for instance, the second hypothesis is rejected based on its p-value, move to the third hypothesis.
                                * The process stops when we retain a null hypothesis. We also keep all subsequent without further testing.
                            ''', mathjax=True),
                html.H5("Hochberg"),
                dcc.Markdown('''
                                Finner test is similar to Holm and also adjusts the value of $$\\alpha$$ in a step-down manner. In the Finner procedure:
                                * Initiate with the most significant p-value.
                                * Reject $$H_1$$ if $$p_1$$ is less than $$1 - (1-\\alpha)^{(k-1)/i}$$. After which, evaluate $$p_2$$ against $$\\alpha / (k-2)$$.
                                * Continue this method sequentially. If, for instance, the second hypothesis is rejected based on its p-value, move to the third hypothesis.
                                * The process stops when we retain a null hypothesis. We also keep all subsequent without further testing.
                            ''', mathjax=True),
                html.H5("Hommel"),
                dcc.Markdown('''
                                The Hommel procedure is acknowledged to be more intricate in terms of both computation and comprehension. Essentially, it requires identifying the greatest value of $$j$$ such that for every $$k$$ ranging from 1 to $$j$$, the condition $$p_{n-j+k} > k\\alpha/j$$. In the case that such a $$j$$ does not exist, we can reject all hypotheses, otherwise we reject all for which $$p_i \\leq \\alpha/j$$.  
                            ''', mathjax=True),
                html.H5("Rom"),
                dcc.Markdown('''
                               Rom test is a modification to Hochberg test to increase its power. This test changes the way alpha is adjusted.
                               $$
                                 \\alpha_{k-i} = \\left[ \\sum_{j=1}^{i-1} \\alpha^j - \\sum_{j=1}^{i-2} \\binom{i}{k} \\alpha^{i-j}_{k-1-j} \\right]/i
                               $$
                            ''', mathjax=True),
                html.H5("Li"),
                dcc.Markdown('''
                                The Li test proposed a two-step rejection procedure:
                                * Step 1: Reject all $H_i$ if $p_{k-1} \\leq \\alpha$. Otherwise, accept the hypothesis associated to $p_{k-1}$ and go to Step 2.
                                * Step 2: Reject any remaining $H_i$ with $p_i \\leq (1-p{k-1})/(1-\\alpha)\\alpha$
                            ''', mathjax=True),
                html.H5("Shaffer"),
                dcc.Markdown('''
                                This test is like Holm's but each p-value associated with the hypothesis  $$H_i$$  is compared as  $$p_i \\leq \\frac{\\alpha}{t_i}$$, where  $$t_i$$
                                is the maximum number of possible hypothesis assuming that the previous  $$(j−1)$$
                                hypothesis have been rejected.
                            ''', mathjax=True),
                html.H5("Nemenyi"),
                dcc.Markdown('''
                                Nemenyi test is similar to the Tukey test for ANOVA and is used when all classifiers are compared to each other. 
                                The performance of two classifiers is significantly different if the corresponding average ranks differ by at least 
                                the critical difference where critical values:
                                $$
                                CD = q_{\\alpha}\\sqrt{\\frac{k(k+1)}{6N}}
                                $$
                            ''', mathjax=True),],
                style={"overflow-y": "scroll", "height": "30em"}  # Ajusta la altura según necesites
            ),
        ]

        return html.Div([
            html.H3(title),
            html.Div(generate_alpha_form(multi_alpha), className="form-element"),
            html.Div([html.H5('Two Groups', style={"display": "inline-block"}),
                      html.Button(html.I(className="fas fa-info-circle"), id="info_pairwise_test", n_clicks=0,
                                  className="button-info"),
                      dbc.Popover(info_pairwise_test, target="info_pairwise_test", body=True, trigger="legacy")],
                     style={"text-align": "center"}),
            html.Div(generate_selector_test(test_two_groups, "selector_two_groups"), className="form-element"),
            html.Div(generate_select_groups(columns, "select_group_1", "First Group"),
                     className="form-element"),
            html.Div(generate_select_groups(columns, "select_group_2", "Second Group"),
                     className="form-element"),
            html.Div([html.H5('Multiple Groups', style={"display": "inline-block"}),
                      html.Button(html.I(className="fas fa-info-circle"), id="info_multiple_test", n_clicks=0,
                                  className="button-info"),
                      dbc.Popover(info_multiple_test, target="info_multiple_test", body=True, trigger="legacy")],
                     style={"text-align": "center"}),
            html.Div(generate_selector_test(test_multiple_groups, "selector_multiple_groups"),
                     className="form-element"),
            html.Div(generate_post_hoc(test_post_hoc), className=class_name_post_hoc),
            html.Div(generate_select_groups(columns, "select_control", "Control"),
                     className="form-element")
        ])


    def create_norm_form(title: str, test_two_groups: list, test_multiple_groups: list, multi_alpha: bool = False):

        info_normality_test = [
            dbc.PopoverHeader(html.H3("Normality Test")),
            dbc.PopoverBody([
                html.P("Data normality suggests that it adheres to a Gaussian (normal) distribution, as represented by its "
                       "probability density function in Equation "),
                dcc.Markdown('''$$ 
                f(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x - \\mu}{\\sigma})^2} $$ ''',
                             mathjax=True),
                html.H5("Shapiro-Wilk"),
                dcc.Markdown('''
                    This test examines the null hypothesis that a population that follows a normal distribution
                    produces a sample $$x_1, x_2, ..., x_n$$. Researchers introduced the Shapiro-Wilk test in 1965. 
                    It calculates a W statistic to determine if a random sample, $$x_1, x_2, ..., x_n$$, originates from a normal distribution.
                    ''',
                             mathjax=True),
                html.H5("D’Agostino-Pearson / Omnibus Tets"),
                dcc.Markdown('''The procedure evaluates the null hypothesis, positing that samples derive from a normally 
                    distributed population. This test combines both the skewness coefficient—which denotes symmetry and typically has 
                    a value of 0 for a normal distribution—and the kurtosis coefficient, which measures peakedness and is usually 0 
                    for a normal distribution. ''', mathjax=True),
                html.H5("Kolmogorov-Smirnov"),
                dcc.Markdown(''' This method adjusts the mean and variance
                    of the benchmark distribution to match the sample’s estimates. However,
                    using these standards to define the reference distribution changes the fundamental distribution of the test’s statistics. Despite these modifications,
                    many studies suggest the test is less precise in identifying normality than the
                    Shapiro–Wilk or D’Agostino-Pearson tests.''', mathjax=True)],
                style={"overflow-y": "scroll", "height": "30em"})]

        info_homoscedasticity_test = [
            dbc.PopoverHeader(html.H3("Homoscedasticity Test")),
            dbc.PopoverBody([
                html.P("Homoscedasticity refers to the assumption that the variances across the data are equal or "
                       "'homogeneous', an essential consideration in parametrics tests."),
                html.H5("Levene"),
                dcc.Markdown('''Levene’s test is a statistical method used for for checking if 
                    multiple samples share the same “spread” or variance. To clarify, if one class’s
                    scores are all over the place and another’s are tightly clustered, your comparisons might be biased. That is where ‘homogeneity of variance’ comes in,
                    ensuring a level playing field. For instance, the analysis of variance (ANOVA)
                    relies on the fundamental assumption that the variances among the different
                    groups or samples are equal. If this assumption does not hold, the ANOVA
                    results might be misleading or inaccurate.''', mathjax=True),
                html.H5("Bartlett"),
                dcc.Markdown(''' Bartlett’s test is used to test if k samples have equal variances. It is more sensitive than Levene to depart from normality. If you have
                    strong evidence that your data do come from a normal, or nearly normal,
                    distribution, then Bartlett’s test has better performance.
                    ''', mathjax=True)],
                style={"overflow-y": "scroll", "height": "30em"})]

        return html.Div([
            html.H3(title),
            html.Div(generate_alpha_form(multi_alpha, switch_selector=False), className="form-element"),
            html.Div([html.H5('Normality', style={"display": "inline-block"}),
                      html.Button(html.I(className="fas fa-info-circle"), id="info_normality_test", n_clicks=0,
                                  className="button-info"),
                      dbc.Popover(info_normality_test, target="info_normality_test", body=True, trigger="legacy")],
                     style={"text-align": "center"}),
            html.Div(generate_selector_test(test_two_groups, "selector_normality"), className="form-element"),
            html.Div([html.H5('Homoscedasticity', style={"display": "inline-block"}),
                      html.Button(html.I(className="fas fa-info-circle"), id="info_homoscedasticity_test", n_clicks=0,
                                  className="button-info"),
                      dbc.Popover(info_homoscedasticity_test, target="info_homoscedasticity_test", body=True, trigger="legacy")],
                     style={"text-align": "center"}),
            html.Div(generate_selector_test(test_multiple_groups, "selector_homoscedasticity"),
                     className="form-element"),
        ])


    def left_block_test(title: str, test_two_groups: list, test_multiple_groups: list, multi_alpha: bool = False):

        return html.Div([
            create_norm_form(title, test_two_groups, test_multiple_groups, multi_alpha),
            html.Div([
                html.Button('Generate Results', id='submit-norm', className="button-form"),
                html.Button('Clear Results', id='reset-norm', className="button-form")
            ], className='form-element')
        ], className='left-block')


    def left_block_experiments(columns: list, title: str, test_two_groups: list, test_multiple_groups: list,
                               test_post_hoc: list = None, user_experiments: list = None, multi_alpha: bool = False):

        table = pd.DataFrame() if user_experiments is None else pd.DataFrame(create_data_table(user_experiments[0],
                                                                                               user_experiments[1]))

        return html.Div([
            create_test_form(columns, title, test_two_groups, test_multiple_groups, test_post_hoc, multi_alpha),
            html.Button('Add Experiment', id='add-experiment', className="button-form center"),
            html.Div([
                html.H3('List of Experiments'),
                html.Div(generate_tabla_experiment(table, height_table="15em"), id="table-experiments"),
                html.Div([
                    html.Div([html.Button("Remove Selected", id="remove-experiment", className="button-form"),
                              html.Button("Remove All", id="remove-all-experiment", className="button-form")]),
                    html.Div([html.Button('Generate Results', id='process-experiment', className="button-form"),
                              html.Button('Clear Results', id='reset-experiment', className="button-form")])
                ], className='form-element'),
            ])
        ], className='left-block')


    def right_block(data: pd.DataFrame, id_block: str = "results_analysis"):
        tabla_data = generate_tabla_of_dataframe(data)
        content_default = html.H3("Load a data set for analysis before selecting the test")
        tabla_data = content_default if tabla_data == "" else tabla_data
        return html.Div([tabla_data, html.Div(children="", className="hidden", id=id_block)],
                        className='right-block', id='right-content')


    def change_page(pathname: str, dataframe: pd.DataFrame, columns: list, user_experiments: list):
        two_groups_test_no_parametrics = [
            {'label': "Wilcoxon", 'value': "Wilcoxon"},
            {'label': "Binomial Sign", 'value': "Binomial Sign"},
            {'label': "Mann-Whitney U", 'value': "Mann-Whitney U"},
        ]

        two_groups_test_parametrics = [
            {'label': "T-Test paired", 'value': "T-Test paired"},
            {'label': "T-Test unpaired", 'value': "T-Test unpaired"}
        ]

        multiple_groups_test_no_parametrics = [
            {'label': "Friedman", 'value': "Friedman"},
            {'label': "Friedman + Iman Davenport", 'value': "Friedman + Iman Davenport"},
            {'label': "Friedman Aligned Ranks", 'value': "Friedman Aligned Ranks"},
            {'label': "Quade", 'value': "Quade"},
            {'label': "Kruskal-Wallis", 'value': "Kruskal-Wallis"},
        ]

        multiple_groups_test_parametrics = [
            {'label': "ANOVA between cases", 'value': "ANOVA between cases"},
            {'label': "ANOVA within cases", 'value': "ANOVA within cases"}
        ]

        post_hoc_no_parametrics = [
            {'label': "Nemenyi", 'value': "Nemenyi"},
            {'label': "Bonferroni", 'value': "Bonferroni"},
            {'label': "Li", 'value': "Li"},
            {'label': "Holm", 'value': "Holm"},
            {'label': "Holland", 'value': "Holland"},
            {'label': "Finner", 'value': "Finner"},
            {'label': "Hochberg", 'value': "Hochberg"},
            {'label': "Hommel", 'value': "Hommel"},
            {'label': "Rom", 'value': "Rom"},
            {'label': "Schaffer", 'value': "Schaffer"}
        ]

        test_normality = [
            {'label': "Shapiro-Wilk", 'value': "Shapiro-Wilk"},
            {'label': "D'Agostino-Pearson", 'value': "D'Agostino-Pearson"},
            {'label': "Kolmogorov-Smirnov", 'value': "Kolmogorov-Smirnov"}
        ]

        test_homocedasticity = [
            {'label': "Levene", 'value': "Levene"},
            {'label': "Bartlett", 'value': "Bartlett"}
        ]

        if pathname in [my_root+"home", my_root]:
            return html.Div(home_page(dataframe))
        elif pathname == my_root + "data_analysis":
            title = "Statistical study"
            return html.Div([
                left_block_experiments(columns, title, two_groups_test_no_parametrics + two_groups_test_parametrics,
                                       multiple_groups_test_no_parametrics + multiple_groups_test_parametrics,
                                       post_hoc_no_parametrics, user_experiments, multi_alpha=True),
                right_block(dataframe, "results_experiments"),
            ], style={'display': 'flex'})
        elif pathname == my_root + "normality_homoscedasticity":
            title = "Normality & Homoscedasticity"
            return html.Div([
                left_block_test(title, test_normality, test_homocedasticity),
                right_block(dataframe, "results_normality")
            ], style={'display': 'flex'})
        else:
            return [html.H1('Página no encontrada'),
                    html.P('La URL ingresada no corresponde a ninguna página.')]


    # Callback para manejar el cambio de vistas basado en la URL
    @app.callback(Output('page-content', 'children'),
                  Input('url', 'pathname'),
                  State('user-data', 'data'),
                  State('user-experiments', 'data'))
    def display_page(pathname, data, user_experiments):
        dataframe = pd.DataFrame(data)
        columns = list(dataframe.columns)
        if all(isinstance(value, str) for value in dataframe[columns]):
            columns = columns[1:]

        return change_page(pathname, dataframe, columns, user_experiments)


    def results_two_groups(data: pd.DataFrame, parameters: dict, alpha: float):
        columns = list(parameters.values())

        if None in columns or columns[1] == columns[2]:
            return

        available_test = {
            "Wilcoxon": no_parametrics.wilcoxon,
            "Binomial Sign": no_parametrics.binomial,
            "Mann-Whitney U": no_parametrics.binomial,
            "T-Test paired": parametrics.t_test_paired,
            "T-Test unpaired": parametrics.t_test_unpaired
        }

        test_function = available_test[columns[0]]

        selected_data = data[columns[1:]]

        statistic, p_value, critical_value, hypothesis = test_function(selected_data, alpha)
        if p_value is None:
            data_results = pd.DataFrame({"Statistic": [statistic], "Critical Value": [critical_value],
                                         "Result": [hypothesis]})
        else:
            data_results = pd.DataFrame({"Statistic": [statistic], "P-value": [p_value], "Result": [hypothesis]})

        caption = f"Results {columns[0]} test (significance level of {alpha})"
        result, test_result_exportable = generate_table_and_textarea(data_results, "7em", caption)

        title = html.H3(f"Two Groups to {columns[1]} vs {columns[2]}", className="title")
        subtitle = html.H5(f"{columns[0]} test (significance level of {alpha})")
        content_to_export = [subtitle, test_result_exportable]
        return html.Div([title, subtitle, result]), html.Div(content_to_export)


    def prueba_paralelizada(args, queue):
        results = args[0](**args[1])
        object_result = []
        if type(results) is pd.DataFrame:
            object_result = [results.to_dict()]
        else:
            if type(results) is tuple and type(results[0]) is pd.DataFrame:
                object_result = [results[0].to_dict()]
            fig = results[-1] if type(results) is tuple else results
            buf = io.BytesIO()  # in-memory files
            fig.savefig(buf, format="png", transparent=True)
            data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
            buf.close()
            object_result.extend(["data:image/png;base64,{}".format(data)])
        queue.put(object_result)


    def generate_table_and_textarea(table_data, height, caption_text, id_val="textarea-dataset"):
        table = generate_tabla_of_dataframe(table_data, height_table=height)
        text_table = utils.dataframe_to_latex(table_data, caption=caption_text)
        textarea = dbc.Textarea(id=id_val, size="lg", value=text_table, style={'height': '200px', "margin-bottom": "0.5em"})
        return table, textarea


    def results_multiple_groups(data: pd.DataFrame, parameters: dict, alpha: float):

        columns = list(parameters.values())

        if columns[0] is None:
            return
        available_test = {"Friedman": no_parametrics.friedman,
                          "Friedman + Iman Davenport": no_parametrics.iman_davenport,
                          "Friedman Aligned Ranks": no_parametrics.friedman_aligned_ranks,
                          "Quade": no_parametrics.quade,
                          "Kruskal-Wallis": no_parametrics.kruskal_wallis,
                          "ANOVA between cases": parametrics.anova_cases,
                          "ANOVA within cases": parametrics.anova_within_cases
                          }

        test_function = available_test[columns[0]]

        parameters_to_function = {"dataset": data, "alpha": alpha, "minimize": parameters["minimize"], "verbose": False,
                                  "apply_correction": True}
        args_functions = inspect.signature(test_function)
        args = {name: parameter.default for name, parameter in args_functions.parameters.items()
                if name not in ["self", "kwargs", "args"]}

        parameters_to_function = {i: parameters_to_function[i] for i in args.keys()}

        if columns[0] == "Kruskal-Wallis":
            statistic, p_value, critical_value, hypothesis = test_function(**parameters_to_function)
            table_results = None
        else:
            table_results, statistic, p_value, critical_value, hypothesis = test_function(**parameters_to_function)

        if p_value is None:
            data_results = pd.DataFrame({"Statistic": [statistic], "Critical Value": [critical_value],
                                         "Result": [hypothesis]})
        else:
            data_results = pd.DataFrame({"Statistic": [statistic], "P-value": [p_value], "Result": [hypothesis]})

        caption = f"Results {columns[0]} test (significance level of {alpha})"
        test_result, test_result_exportable = generate_table_and_textarea(data_results, "7em", caption)
        test_subtitle = html.H5(f"{columns[0]} test (significance level of {alpha})")
        title = html.H3(f"Multiple Groups", className="title")
        content = [title, test_subtitle, test_result]
        content_to_export = [test_subtitle, test_result_exportable]
        rankings_with_label = {}
        if type(table_results) is list:
            tab_1, tab_1_exportable = generate_table_and_textarea(table_results[0], "14em", "Data Summary")
            caption_2 = f"Summary {columns[0]} test (significance level of {alpha})"
            tab_2, tab_2_exportable = generate_table_and_textarea(table_results[1], "14em", caption_2)

            content.extend([tab_1, tab_2])
            content_to_export.extend([tab_1_exportable, tab_2_exportable])
        elif type(table_results) is dict:
            rankings_data = {i[0]: [round(i[1], 5)] for i in table_results.items()}
            rankings_data = pd.DataFrame(rankings_data)
            rankings_with_label = table_results
            caption = f"Rankings of {columns[0]} test (significance level of {alpha})"
            table, table_exportable = generate_table_and_textarea(rankings_data, "7.2em", caption)

            content.append(table)
            content_to_export.append(table_exportable)

        if not (columns[1] is None) and type(table_results) is dict:
            available_post_hoc = {"Nemenyi": no_parametrics.nemenyi,
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
            post_hoc_function = available_post_hoc[columns[1]]

            parameters_to_function = {"ranks": rankings_with_label, "num_cases": data.shape[0], "alpha": alpha,
                                      "minimize": parameters["minimize"], "verbose": False, "name_fig": "",
                                      "control": parameters["control"], "type_rank": columns[0]
                                      }
            args_functions = inspect.signature(post_hoc_function)
            args = {name: parameter.default for name, parameter in args_functions.parameters.items()
                    if name not in ["self", "kwargs", "args"]}

            parameters_to_function = {i: parameters_to_function[i] for i in args.keys()}

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=prueba_paralelizada, args=([post_hoc_function, parameters_to_function],
                                                                                result_queue))
            process.start()
            process.join()
            results = result_queue.get()

            text_post_hoc = f"Post hoc: {columns[1]} test (significance level of {alpha})"

            if columns[1] == "Nemenyi" and alpha in [0.025, 0.005, 0.001]:
                text_post_hoc = text_post_hoc[:-1] + " with approximate results)"

            post_hoc_subtitle = html.H5(text_post_hoc,
                                        style={"margin-top": "0.5em"})

            if type(results[0]) is dict:
                post_hoc_result = [generate_tabla_of_dataframe(pd.DataFrame(results[0]))]
                text_table = utils.dataframe_to_latex(pd.DataFrame(results[0]),
                                                      caption=f"Post hoc: {columns[1]} test (significance level of {alpha})"
                                                      )
                table_exportable = dbc.Textarea(id="textarea-dataset", size="lg", value=text_table,
                                                style={'height': '200px'})
                content_to_export.extend([post_hoc_subtitle, table_exportable])
                post_hoc_result.extend([html.Img(src=results[-1], width="50%", height="50%")])
                post_hoc_result = html.Div(post_hoc_result)
            else:
                post_hoc_result = html.Img(src=results[-1], width="100%", height="100%")
            content.extend([post_hoc_subtitle, post_hoc_result])

        return [html.Div(children=content), html.Div(content_to_export)]


    def create_data_table(names_exp, parameters_exp):
        table = {"Name Experiment": [], "Alpha": [], "Test": [], "Criterion": []}
        for index, value in enumerate(zip(names_exp, parameters_exp)):
            experiment, parameter = value
            table["Name Experiment"].append(experiment)
            table["Alpha"].append(parameter["alpha"])
            criterion = "" if "minimize" not in parameter.keys() else "Maximize" if parameter["minimize"] else "Minimize"
            table["Criterion"].append(criterion)
            if "post_hoc" in parameter.keys():
                # test = [parameter["test"], parameter["post_hoc"]]
                test = f"{parameter['test']}"
                if not (parameter["post_hoc"] is None):
                    test += f", {parameter['post_hoc']}"
                    if "control" in parameter.keys() and not (parameter["control"] is None):
                        test += f" ({parameter['control']})"
            else:
                test = f"{parameter['test']} ({parameter['first_group']}, {parameter['second_group']})"
            table["Test"].append(test)

        return table


    def generate_tabla_experiment(df: pd.DataFrame, height_table: str = '30em'):
        if df is None or df.empty:
            df = pd.DataFrame({"Name Experiment": [], "Alpha": [], "Test": []})
        return dag.AgGrid(
            id="list-experiment",
            rowData=df.to_dict("records"),
            columnDefs=[
                {'headerName': col, 'field': col} for col in df.columns
            ],
            defaultColDef={"resizable": True, "filter": False, "minWidth": 100, "floatingFilter": False},
            columnSize="responsiveSizeToFit",
            style={'height': height_table},
            dashGridOptions={"rowSelection": "multiple", "rowMultiSelectWithClick": True},
        )


    def generate_inform(dataset, experiments: dict, generate_pdf: bool = False, name_pdf: str = "informe.pdf"):
        utils.analysis_of_experiments(dataset, experiments, generate_pdf, name_pdf)


    @app.callback(Output('results_experiments', 'children'),
                  Output('results_experiments', 'className'),
                  Output("reset-experiment", "n_clicks"),
                  Output("modal-export", "children"),
                  Output("user-file", "data"),
                  Input("process-experiment", "n_clicks"),
                  Input("reset-experiment", "n_clicks"),
                  State('user-experiments', 'data'),
                  State('user-data', 'data'))
    def process_experiment(n_clicks, reset, user_experiments, current_session):
        if n_clicks is None and reset is None:
            return dash.no_update

        name_file_pdf = ""
        generate_pdf = True
        export_modal = [dbc.ModalHeader(dbc.ModalTitle("Export Results")),
                        dbc.ModalBody("Here, the results of all the tables generated by the experiments will be displayed.")
                       ]
        if reset:
            return html.Div(""), "", None, export_modal, name_file_pdf

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
        parametrics_test = ["T-Test paired", "T-Test unpaired", "ANOVA between cases", "ANOVA within cases"]

        names_experiments, parameter_experiments = user_experiments[0], user_experiments[1]
        df = pd.DataFrame(current_session)
        experiments = utils.generate_json(names_experiments, parameter_experiments)
        if generate_pdf:
            current_datetime = datetime.now()
            current_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
            name_file_pdf = f"inform_{current_datetime}.pdf"
            args = (df, experiments, generate_pdf, name_file_pdf)
            process = multiprocessing.Process(target=generate_inform, args=args)
            process.start()
            process.join()

        content = []
        content_exportable = [dbc.ModalHeader(dbc.ModalTitle("Export Results"))]
        for name_experiment in experiments.keys():
            info_experiment = experiments[name_experiment]
            type_test = "parametric" if info_experiment["test"] in parametrics_test else "no parametrics"
            alpha_results = []
            if info_experiment["test"] in available_test_two_groups.keys():
                test_selected = {"selected_test": info_experiment["test"], "group_1": info_experiment["first_group"],
                                 "group_2": info_experiment["second_group"]}
                if type(info_experiment["alpha"]) is list:
                    alpha_results = [results_two_groups(df, test_selected, i) for i in info_experiment["alpha"]]
                else:
                    alpha_results = [results_two_groups(df, test_selected, info_experiment["alpha"])]

                alpha_results, table_results = zip(*alpha_results)

                table_results = (html.H3(f"Experiment {name_experiment} : {type_test}",
                                         className="title"),) + table_results
                table_results += (dbc.Textarea(id="textarea-dataset", size="sm", value="", style={'height': '0',
                                                                                                  'visibility': 'hidden'}),)
                content_exportable.extend([dbc.ModalBody(table_results)])
            elif info_experiment["test"] in available_test_multiple_groups.keys():
                test_selected = {"test": info_experiment["test"], "post_hoc": info_experiment["post_hoc"],
                                 "minimize": info_experiment["minimize"],
                                 "control": info_experiment["control"] if info_experiment["post_hoc"] not in ["Nemenyi",
                                                                                                              "Schaffer"]
                                 else None}

                if type(info_experiment["alpha"]) is list:
                    alpha_results = [results_multiple_groups(df, test_selected, i) for i in info_experiment["alpha"]]
                    alpha_results, table_results = zip(*alpha_results)
                else:
                    alpha_results = [results_multiple_groups(df, test_selected, info_experiment["alpha"])]
                    alpha_results, table_results = zip(*alpha_results)

                table_results = (html.H3(f"Experiment {name_experiment} : {type_test}",
                                         className="title"), ) + table_results
                table_results += (dbc.Textarea(id="textarea-dataset", size="sm", value="", style={'height': '0',
                                                                                                  'visibility': 'hidden'}),)
                content_exportable.extend([dbc.ModalBody(table_results)])

            # content.extend([html.H3(f"Experiment {name_experiment} : {type_test}", className="title")])
            # content.extend(alpha_results)
            # content.extend(html.Hr())
            content.extend([dbc.AccordionItem(children=html.Div(alpha_results),
                                              title=f"Experiment {name_experiment} : {type_test}")])
        show_info = html.Div([html.Button("Download Report", id="btn-download-txt", className="button-form center"),
                              dbc.Accordion(content, id="accordion-always-open", always_open=True)])
        return show_info, "", None, html.Div(content_exportable), name_file_pdf


    def get_letter_experiment(index):
        letters = []
        while index > 0:
            index, remainder = divmod(index - 1, 26)
            letters.append(chr(65 + remainder))
        return ''.join(reversed(letters))


    def get_number_from_text(text):
        result = 0
        for char in text:
            result = result * 26 + (ord(char) - 64)
        return result


    @app.callback(Output("table-experiments", "children"),
                  Output('user-experiments', 'data'),
                  Output("add-experiment", "n_clicks"),
                  Output("remove-experiment", "n_clicks"),
                  Output("remove-all-experiment", "n_clicks"),
                  Input("add-experiment", "n_clicks"),
                  Input("remove-experiment", "n_clicks"),
                  Input("remove-all-experiment", "n_clicks"),
                  State('selector_alpha', 'value'),
                  State('selector_two_groups', 'value'),
                  State('select_group_1', 'value'),
                  State('select_group_2', 'value'),
                  State('selector_multiple_groups', 'value'),
                  State('selector_post_hoc', 'value'),
                  State('select_control', 'value'),
                  State('user-experiments', 'data'),
                  State("list-experiment", "selectedRows"),
                  State("criterion_switch", "on")
                  )
    def add_experiment(add_n_clicks, remove_n_clicks, remove_all_n_clicks, alpha, test_two_groups, group_1, group_2,
                       test_multiple_groups, test_post_hoc, control, user_experiments, selected_experiment, criterion):
        if add_n_clicks is None and remove_n_clicks is None and remove_all_n_clicks is None:
            return dash.no_update

        if add_n_clicks:
            names_exp, parameters_exp = [], []

            if not (user_experiments is None):
                names_exp, parameters_exp = user_experiments[0], user_experiments[1]

            # Check Available test
            index_exp = 0 if len(names_exp) == 0 else get_number_from_text(names_exp[-1])
            index_exp += 1
            if None not in [test_two_groups, group_1, group_2] and group_1 != group_2:
                names_exp.append(get_letter_experiment(index_exp))
                parameters_exp.append({"alpha": alpha, "test": test_two_groups, "first_group": group_1,
                                       "second_group": group_2})
                index_exp += 1

            if not (test_multiple_groups is None):
                names_exp.append(get_letter_experiment(index_exp))
                args = {"alpha": alpha, "test": test_multiple_groups, "post_hoc": test_post_hoc, "minimize": criterion}
                if test_post_hoc not in ["Nemenyi", "Schaffer"]:
                    args["control"] = control

                parameters_exp.append(args)

            tabla = create_data_table(names_exp, parameters_exp)

            new_user_experiments = [names_exp, parameters_exp]

        elif remove_all_n_clicks:
            tabla = pd.DataFrame()
            new_user_experiments = [[], []]

        elif selected_experiment:
            names_exp, parameters_exp = user_experiments[0], user_experiments[1]
            index_to_remove = [i["Name Experiment"] for i in selected_experiment]
            experiments = [[index, value] for index, value in zip(names_exp, parameters_exp) if
                           str(index) not in index_to_remove]
            if experiments:
                names_exp, parameters_exp = zip(*experiments)
            else:
                names_exp, parameters_exp = [], []

            tabla = create_data_table(names_exp, parameters_exp)
            new_user_experiments = [names_exp, parameters_exp]

        else:
            return dash.no_update

        return generate_tabla_experiment(pd.DataFrame(tabla), height_table="15em"), new_user_experiments, None, None, None


    def results_normality(dataset: pd.DataFrame, alpha: float, test_normality: str, test_homoscedasticity: str):
        available_normality_test = {"Shapiro-Wilk": normality.shapiro_wilk_normality,
                                    "D'Agostino-Pearson": normality.d_agostino_pearson,
                                    "Kolmogorov-Smirnov": normality.kolmogorov_smirnov,
                                    }
        available_homocedasticity_test = {"Levene": homoscedasticity.levene_test,
                                          "Bartlett": homoscedasticity.bartlett_test}
        content, content_exportable = [], []
        alpha = float(alpha)
        if test_normality:
            if test_normality not in available_normality_test:
                raise ValueError(f"Test de normalidad '{test_normality}' no disponible.")
            test_normality_function = available_normality_test[test_normality]
            columns = list(dataset.columns)
            statistic_list, p_value_list, cv_value_list, hypothesis_list, graficos = [], [], [], [], []
            for i in range(1, len(columns)):
                statistic, p_value, cv_value, hypothesis = test_normality_function(dataset[columns[i]].to_numpy(), alpha)
                statistic_list.append(statistic)
                p_value_list.append(p_value)
                cv_value_list.append(cv_value)
                hypothesis_list.append(hypothesis)
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(target=prueba_paralelizada, args=([normality.qq_plot,
                                                                                     {"data": dataset[
                                                                                         columns[i]].to_numpy()}],
                                                                                    result_queue))
                process.start()
                process.join()
                qq_plot_figure = result_queue.get()

                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(target=prueba_paralelizada, args=([normality.pp_plot,
                                                                                     {"data": dataset[
                                                                                         columns[i]].to_numpy()}],
                                                                                    result_queue))
                process.start()
                process.join()
                pp_plot_figure = result_queue.get()

                graficos.append([html.Img(src=qq_plot_figure[-1], width="50%", height="50%"),
                                 html.Img(src=pp_plot_figure[-1], width="50%", height="50%")])

            test_subtitle = html.H5(f"{test_normality} test (significance level of {alpha})")
            title = html.H3(f"Normality Analysis", className="title")

            results_test = pd.DataFrame({"Dataset": columns[1:], "Statistic": statistic_list, "p-value": p_value_list,
                                         "Results": hypothesis_list})

            table, export_table = generate_table_and_textarea(results_test, height=f"{4.2 * len(results_test)}em",
                                                              caption_text=test_subtitle)

            images = html.Div([
                html.Button("Show Graphs pp-plot and qq-plot", id="collapse-button", n_clicks=0,
                            style={"margin-top": "0.5em", "margin-bottom": "0.5em"}),
                dbc.Collapse(dbc.Card(dbc.CardBody([html.Div([html.H3(columns[i+1]), html.Div([fig[0], fig[1]])]) for i, fig
                                                    in enumerate(graficos)])),
                             id="collapse",
                             is_open=False,
                             )
            ])
            content = [title, test_subtitle, table, images]
            content_exportable = [html.H3(f"{test_normality} test (significance level of {alpha})", className="title"),
                                  export_table]
            content_exportable += (dbc.Textarea(id="textarea-dataset", size="sm", value="", style={'height': '0',
                                                                                                   'visibility': 'hidden'}),)

        if not (test_homoscedasticity is None):
            test_homocedasticity_function = available_homocedasticity_test[test_homoscedasticity]
            statistic, p_value, cv_value, hypothesis = test_homocedasticity_function(dataset, alpha)
            title_homoscedasticity = f"{test_homoscedasticity} test (significance level of {alpha})"
            if p_value is None:
                test_result = pd.DataFrame({"Statistic": [statistic], "Critical Value": [cv_value], "Result": [hypothesis]})
            else:
                test_result = pd.DataFrame({"Statistic": [statistic], "P-value": [p_value], "Result": [hypothesis]})

            table, export_table = generate_table_and_textarea(test_result, height=f"{4.2 * len(test_result)}em",
                                                              caption_text=title_homoscedasticity)

            test_subtitle = html.H5(title_homoscedasticity)
            title = html.H3(f"Homocedasticity Analysis", className="title")

            content.extend([title, test_subtitle, table])
            homoscedasticity_export = [html.H3(title_homoscedasticity, className="title"), export_table,
                                       dbc.Textarea(id="textarea-dataset", size="sm", value="",
                                                    style={'height': '0', 'visibility': 'hidden'})]
            content_exportable.extend(homoscedasticity_export)

        return content, content_exportable


    @app.callback(Output('results_normality', 'children'),
                  Output('results_normality', 'className'),
                  Output("reset-norm", "n_clicks"),
                  Output("body-modal-export", "children"),
                  Input("submit-norm", "n_clicks"),
                  Input("reset-norm", "n_clicks"),
                  State('selector_alpha', 'value'),
                  State('selector_normality', 'value'),
                  State('selector_homoscedasticity', 'value'),
                  State('user-data', 'data'))
    def process_normality(n_clicks, reset, alpha, test_normality, test_homoscedasticity, current_data):
        if n_clicks is None and reset is None:
            return dash.no_update
        if reset:
            return html.Div(""), "", None

        dataset = pd.DataFrame(current_data)
        content, content_export = results_normality(dataset, alpha, test_normality, test_homoscedasticity)

        return html.Div(content), "", None, html.Div(content_export)


    @app.callback(
        Output("collapse", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("collapse", "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open


    @app.callback(
        Output("download-text", "data"),
        Input("btn-download-txt", "n_clicks"),
        State("user-file", "data")
    )
    def download_report(n_clicks, name_file):
        if n_clicks is None:
            return dash.no_update
        return dcc.send_file(name_file)


    def start_app(host: str = '127.0.0.1', port: int = 8050):
        app.run(debug=False, port=port, host=host)


    def get_app():
        return app


    if __name__ == '__main__':
        app.run(host="0.0.0.0", port=9050, debug=False)
