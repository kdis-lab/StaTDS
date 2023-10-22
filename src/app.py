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

import multiprocessing

#TODO Check imports

import no_parametrics, parametrics, utils


external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/app/style.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
# suppress_callback_exceptions=True Esto no es una buena práctica pero es la única forma de mantener el control dinámico


with open("assets/app/README.md", 'r') as film:
    readme_content = film.read()


def generate_text_sample():
    with open("assets/app/sample_dataset.csv", "r") as f:
        text = f.read()
    return text


def generate_navigator_menu():
    menus = [
                {"title": html.Img(src="assets/images/logo.png", style={"width": "5em", "height": "1em"}),
                 "href": "home", "icon": "mdi:home-outline"},
                {"title": "Data Analysis", "href": "data_analysis"},
                {"title": "Normality & Homoscedasticity", "href": "normality_homoscedasticity"},
            ]
    import_button = dbc.Button(children="Import Data", id="import-Data", outline=True, className="menu item-menu")
    export_button = dbc.Button(children="Export Results", id="export-data", outline=True, className="menu item-menu")
    content = [dcc.Link(children=[html.Div(item["title"], className="item-menu")],
                        href=item["href"] if "href" in item.keys() else None, className="menu") for item in menus]

    download_file = dcc.Download(id="download-text")
    content = content + [import_button, export_button, generate_import_data_page(), generate_export_table_page(),
                         download_file]
    return html.Div(
        children=content, className="navigator_menu"
    )


def generate_import_data_page():
    example_of_dataset = generate_text_sample()

    modal = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Import Dataset to analysis")),
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
                                            style={'height': '200px'})]),
                dbc.ModalBody(["Upload File:",
                               dbc.Textarea(id="path_of_file", size="sm", readOnly=True,
                                            style={'resize': 'none', 'font-size': '1em', 'height': "1em"}),
                               dcc.Upload(
                                   id='upload-data',
                                   children=html.Div([
                                       'Drag and Drop or ',
                                       html.A('Select File')
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
                dbc.ModalHeader(dbc.ModalTitle("Export Results"))
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

    return dag.AgGrid(
        id="table",
        rowData=df.to_dict("records"),
        columnDefs=[
            {'headerName': col, 'field': col} for col in df.columns
        ],
        defaultColDef={"resizable": True, "filter": False, "minWidth": 150, "floatingFilter": False},
        columnSize="sizeToFit",
        style={'height': height_table}
    )


def generate_home_page(dataframe: pd.DataFrame):
    global readme_content

    main_content = html.Div(
        children=[
            html.H1(children='Statistical Test App', className="title-app"),
            html.Div(id="table-content", children=generate_tabla_of_dataframe(dataframe),
                     className="table-info hidden" if dataframe is None or dataframe.empty else "table-info"),
            html.Div(children=[html.P(readme_content)],
                     className="content-info")
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
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


def dataframe_to_text(df: pd.DataFrame):
    column_names = df.columns.tolist()
    data_rows = [",".join(map(str, row)) for _, row in df.iterrows()]

    text = "\n".join([",".join(column_names)] + data_rows)

    return text


@app.callback(Output('path_of_file', 'value'),
              Output('textarea-dataset', 'value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('select-separator', 'value'))
def update_output(list_of_contents, list_of_names, separator):
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names, separator)
        return [list_of_names, dataframe_to_text(df)]

    return "", ""


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
    if n_clicks is None:
        return dash.no_update

    if n_clicks:
        dataframe = pd.read_csv(io.StringIO(textarea_value), sep=separator)
        return None, dataframe.to_dict(), n_clicks_modal + 1, "/"

    return None, current_session, n_clicks_modal, "/"


def generate_alpha_form(multiple_alpha: bool = False):
    available_alpha = [0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
    title = "Alpha(s)" if multiple_alpha else "Alpha"
    return [html.Label(title, style={"margin-left": "1em", "margin-right": "1em"}),
            dcc.Dropdown(
                    options=[{'label': str(i), 'value': str(i)} for i in available_alpha],
                    value=str(available_alpha[1]), style={'width': 'auto', "min-width": "8em", "max-width": "16em"},
                    id="selector_alpha", multi=multiple_alpha),
            html.Div([
                html.Label('Generate Reports'),
                daq.BooleanSwitch(id='reports_switch', on=False)
            ], className="slider-content")]


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
    return html.Div([
        html.H3(title),
        html.Div(generate_alpha_form(multi_alpha), className="form-element"),
        html.H5('Two Groups'),
        html.Div(generate_selector_test(test_two_groups, "selector_two_groups"), className="form-element"),
        html.Div(generate_select_groups(columns, "select_group_1", "First Group"),
                 className="form-element"),
        html.Div(generate_select_groups(columns, "select_group_2", "Second Group"),
                 className="form-element"),
        html.H5('Multiple Groups'),
        html.Div(generate_selector_test(test_multiple_groups, "selector_multiple_groups"),
                 className="form-element"),
        html.Div(generate_post_hoc(test_post_hoc), className=class_name_post_hoc),
        html.Div(generate_select_groups(columns, "select_control", "Control"),
                 className="form-element"),
    ])


def create_norm_form(title: str, test_two_groups: list, test_multiple_groups: list, multi_alpha: bool = False):
    return html.Div([
        html.H3(title),
        html.Div(generate_alpha_form(multi_alpha), className="form-element"),
        html.H5('Normality'),
        html.Div(generate_selector_test(test_two_groups, "selector_normality"), className="form-element"),
        html.H5('Homoscedasticity'),
        html.Div(generate_selector_test(test_multiple_groups, "selector_homoscedasticity"),
                 className="form-element"),
    ])


def left_block_test(title: str, test_two_groups: list, test_multiple_groups: list, multi_alpha: bool = False):

    return html.Div([
        create_norm_form(title, test_two_groups, test_multiple_groups, multi_alpha),
        html.Div([
            html.Button('Send', id='submit-norm'),
            html.Button('Reset', id='reset-norm')
        ], className='form-element')
    ], className='left-block')


def left_block_experiments(columns: list, title: str, test_two_groups: list, test_multiple_groups: list,
                           test_post_hoc: list = None, user_experiments: list = None, multi_alpha: bool = False):

    table = pd.DataFrame() if user_experiments is None else pd.DataFrame(create_data_table(user_experiments[0],
                                                                                           user_experiments[1]))

    return html.Div([
        create_test_form(columns, title, test_two_groups, test_multiple_groups, test_post_hoc, multi_alpha),
        html.Div([
            html.Button('Add Experiment', id='add-experiment'),
            html.Div([html.Button('Send', id='process-experiment'),
                      html.Button('Reset', id='reset-experiment', style={"margin-left": "0.5em"})])
        ], className='form-element'),
        html.Div([
            html.H3('List of Experiments'),
            html.Div(generate_tabla_experiment(table, height_table="15em"), id="table-experiments"),
            html.Div([
                html.Button("Remove Seleted Experiments", id="remove-experiment"),
                html.Button("Remove All Experiments", id="remove-all-experiment")
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
        {'label': "Friedman Aligned Ranks", 'value': "Friedman Aligned Ranks"},
        {'label': "Quade", 'value': "Quade"},
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

    if pathname in ['/home', "/"]:
        return html.Div(home_page(dataframe))
    elif pathname == "/data_analysis":
        title = "Statistical study"
        return html.Div([
            left_block_experiments(columns, title, two_groups_test_no_parametrics + two_groups_test_parametrics,
                                   multiple_groups_test_no_parametrics + multiple_groups_test_parametrics,
                                   post_hoc_no_parametrics, user_experiments, multi_alpha=True),
            right_block(dataframe, "results_experiments"),
        ], style={'display': 'flex'})
    elif pathname == "/normality_homoscedasticity":
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
        "Wilcoxon": no_parametrics.wilconxon,
        "Binomial Sign": no_parametrics.binomial,
        "Mann-Whitney U": no_parametrics.binomial,
        "T-Test paired": parametrics.t_test_paired,
        "T-Test unpaired": parametrics.t_test_unpaired
    }

    test_function = available_test[columns[0]]

    selected_data = data[columns[1:]]

    statistic, critical_value, p_value, hypothesis = test_function(selected_data, alpha)
    if p_value is None:
        result = f"Statistic: {statistic} Critical Value: {critical_value} Result: {hypothesis}"
    else:
        result = f"Statistic: {statistic} P-value: {p_value} Result: {hypothesis}"
    title = html.H3(f"Two Groups to {columns[1]} vs {columns[2]}", className="title")
    subtitle = html.H5(f"{columns[0]} test (significance level of {alpha})")
    return html.Div([title, subtitle, result])


def prueba_paralelizada(args, queue):
    # TODO Cambiar el nombre de esta función y los argumentos
    results = args[0](**args[1])
    object_result = []
    if type(results) is pd.DataFrame:
        object_result = [results.to_dict()]
    else:
        if type(results[0]) is pd.DataFrame:
            object_result = [results[0].to_dict()]
        fig = results[-1]
        buf = io.BytesIO()  # in-memory files
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
        buf.close()
        object_result.extend(["data:image/png;base64,{}".format(data)])
    queue.put(object_result)


def results_multiple_groups_ant(data: pd.DataFrame, parameters: dict, alpha: float):
    # TODO Cambiar esto cuando este el anova mejorado:

    def anova_cases(dataset: pd.DataFrame, alpha_value: float = 0.05):
        s, p, c, h = parametrics.anova_test(dataset, alpha_value)
        return pd.DataFrame(),  s, p, c, h

    def anova_within_cases(dataset: pd.DataFrame, alpha_value: float = 0.05):
        s, p, c, h = parametrics.anova_within_cases_test(dataset, alpha_value)
        return pd.DataFrame(),  s, p, c, h

    columns = list(parameters.values())
    # Se genera mediante la librería
    if columns[0] is None:
        return

    available_test = {"Friedman": no_parametrics.friedman,
                      "Friedman Aligned Ranks": no_parametrics.friedman_aligned_ranks,
                      "Quade": no_parametrics.quade,
                      "ANOVA between cases": anova_cases,
                      "ANOVA within cases": anova_within_cases
                      }

    test_function = available_test[columns[0]]

    rankings_with_label, statistic, p_value, critical_value, hypothesis = test_function(data, alpha)
    test_result = f"Statistic: {statistic} P-value: {p_value} Result: {hypothesis}"
    test_subtitle = html.H5(f"{columns[0]} test (significance level of {alpha})")
    title = html.H3(f"Multiple Groups", className="title")

    rankings = pd.DataFrame({i[0]: [round(i[1], 5)] for i in rankings_with_label.items()})
    table = generate_tabla_of_dataframe(rankings, height_table="7.2em")
    content = [title, test_subtitle, test_result, table]
    if not(columns[1] is None):
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
                                  "criterion": False, "verbose": False, "name_fig": "",
                                  "all_vs_all": True, "control": None}
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
        if type(results[0]) is dict:
            post_hoc_result = [generate_tabla_of_dataframe(pd.DataFrame(results[0]))]
            post_hoc_result.extend([html.Img(src=results[-1], width="50%", height="50%")])
            post_hoc_result = html.Div(post_hoc_result)
        else:
            post_hoc_result = html.Img(src=results[-1], width="100%", height="100%")
        text_post_hoc = f"Post hoc: {columns[1]} test (significance level of {alpha})"
        if columns[1] == "Nemenyi" and alpha in ["0.025", "0.005", "0.001"]:
            text_post_hoc = text_post_hoc[:-1] + "approximate results)"
        
        post_hoc_subtitle = html.H5(text_post_hoc,
                                    style={"margin-top": "0.5em"})
        content.extend([post_hoc_subtitle, post_hoc_result])
    return html.Div(children=content)

def generate_table_and_textarea(table_data, height, caption_text, id_val="textarea-dataset"):
    table = generate_tabla_of_dataframe(table_data, height_table=height)
    text_table = utils.dataframe_to_latex(table_data, caption=caption_text)
    textarea = dbc.Textarea(id=id_val, size="lg", value=text_table, style={'height': '200px'})
    return table, textarea

def results_multiple_groups(data: pd.DataFrame, parameters: dict, alpha: float):
    # TODO Cambiar esto cuando este el anova mejorado:

    def anova_cases(dataset: pd.DataFrame, alpha_value: float = 0.05):
        s, p, c, h = parametrics.anova_test(dataset, alpha_value)
        return pd.DataFrame(), s, p, c, h

    def anova_within_cases(dataset: pd.DataFrame, alpha_value: float = 0.05):
        s, p, c, h = parametrics.anova_within_cases_test(dataset, alpha_value)
        return pd.DataFrame(), s, p, c, h

    columns = list(parameters.values())
    # Se genera mediante la librería
    if columns[0] is None:
        return

    available_test = {"Friedman": no_parametrics.friedman,
                      "Friedman Aligned Ranks": no_parametrics.friedman_aligned_ranks,
                      "Quade": no_parametrics.quade,
                      "ANOVA between cases": parametrics.anova_cases,
                      "ANOVA within cases": parametrics.anova_within_cases
                      }

    test_function = available_test[columns[0]]

    table_results, statistic, p_value, critical_value, hypothesis = test_function(data, alpha)

    if p_value is None:
        test_result = f"Statistic: {statistic} Critical Value: {critical_value} Result: {hypothesis}"
    else:
        test_result = f"Statistic: {statistic} P-value: {p_value} Result: {hypothesis}"
    test_subtitle = html.H5(f"{columns[0]} test (significance level of {alpha})")
    title = html.H3(f"Multiple Groups", className="title")
    content = [title, test_subtitle, test_result]
    content_to_export = [test_subtitle]
    if type(table_results) is list:
        tab_1, tab_1_exportable = generate_table_and_textarea(table_results[0], "14em", "Data Summary")
        caption_2 = f"Summary {columns[0]} test (significance level of {alpha})"
        tab_2, tab_2_exportable = generate_table_and_textarea(table_results[1], "14em", caption_2)
        
        content.extend([tab_1, tab_2])
        content_to_export.extend([tab_1_exportable, tab_2_exportable])
    else:
        rankings_data = {i[0]: [round(i[1], 5)] for i in table_results.items()}
        caption = f"Rankings of {columns[0]} test (significance level of {alpha})"
        table, table_exportable = generate_table_and_textarea(rankings_data, "7.2em", caption)
        
        content.append(table)
        content_to_export.append(table_exportable)

    if not (columns[1] is None):
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
                                  "criterion": False, "verbose": False, "name_fig": "", "control": parameters["control"]
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


def generate_analysis(test_selected: dict):
    result_two_groups = results_two_groups(test_selected["data"], test_selected["two_groups"], test_selected["alpha"])
    result_multiple_groups = results_multiple_groups(test_selected["data"], test_selected["multiple_groups"],
                                                     test_selected["alpha"])
    content = [result_two_groups, result_multiple_groups[0]]
    """if test_selected["inform"] is True:
        content.extend([html.Button("Download Text", id="btn-download-txt"),
                        ])"""

    return html.Div(content)


def create_data_table(names_exp, parameters_exp):
    table = {"Name Experiment": [], "Alpha": [], "Test": []}
    for index, value in enumerate(zip(names_exp, parameters_exp)):
        experiment, parameter = value
        table["Name Experiment"].append(experiment)
        table["Alpha"].append(parameter["alpha"])
        if "post_hoc" in parameter.keys():
            # test = [parameter["test"], parameter["post_hoc"]]
            test = f"{parameter['test']}"
            if not(parameter["post_hoc"] is None):
                test += f", {parameter['post_hoc']}"
                if "control" in parameter.keys() and not(parameter["control"] is None):
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
        defaultColDef={"resizable": True, "filter": False, "minWidth": 150, "floatingFilter": False},
        columnSize="sizeToFit",
        style={'height': height_table},
        dashGridOptions={"rowSelection": "multiple", "rowMultiSelectWithClick": True},
    )


def generate_inform(dataset, experiments: dict, generate_pdf: bool = False, name_pdf: str = "informe.pdf"):
    utils.analysis_of_experiments(dataset, experiments, generate_pdf, name_pdf)


@app.callback(Output('results_experiments', 'children'),
              Output('results_experiments', 'className'),
              Output("reset-experiment", "n_clicks"),
              Output("modal-export", "children"),
              Output("download-text", "data"),
              Input("process-experiment", "n_clicks"),
              Input("reset-experiment", "n_clicks"),
              State('reports_switch', 'on'),
              State('user-experiments', 'data'),
              State('user-data', 'data'))
def process_experiment(n_clicks, reset, generate_pdf, user_experiments, current_session):
    if n_clicks is None and reset is None:
        return dash.no_update

    name_file_pdf = ""

    if reset:
        return html.Div(""), "", None, dbc.ModalHeader(dbc.ModalTitle("Export Results")), name_file_pdf

    available_test_multiple_groups = {"Friedman": no_parametrics.friedman,
                                      "Friedman Aligned Ranks": no_parametrics.friedman_aligned_ranks,
                                      "Quade": no_parametrics.quade,
                                      "ANOVA between cases": parametrics.anova_test,
                                      "ANOVA within cases": parametrics.anova_within_cases_test
                                      }
    available_test_two_groups = {"Wilcoxon": no_parametrics.wilconxon,
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
        elif info_experiment["test"] in available_test_multiple_groups.keys():
            test_selected = {"test": info_experiment["test"], "post_hoc": info_experiment["post_hoc"],
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

        content.extend([html.H3(f"Experiment {name_experiment} : {type_test}", className="title")])
        content.extend(alpha_results)

    if generate_pdf is True:
        content.extend([html.Button("Download Text", id="btn-download-txt"),
                        dcc.Download(id="download-text")])
        # TODO Poner que el nombre sea aleatorio con respecto al tiempo para que no haya solapamientos  

    send_file = None if name_file_pdf == "" else dcc.send_file(name_file_pdf)
    return html.Div(content), "", None, html.Div(content_exportable), send_file


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
              State("list-experiment", "selectedRows")
              )
def add_experiment(add_n_clicks, remove_n_clicks, remove_all_n_clicks, alpha, test_two_groups, group_1, group_2,
                   test_multiple_groups, test_post_hoc, control, user_experiments, selected_experiment):
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
            args = {"alpha": alpha, "test": test_multiple_groups, "post_hoc": test_post_hoc, "criterion": True}
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


# TODO FALTA ARREGLAR LAS FUNCIONES DE LOS TEST PARAMETRICOS
def results_normality(dataset: pd.DataFrame, alpha: float, test_normality: str, test_homoscedasticity: str):
    available_normality_test = {"Shapiro-Wilk": parametrics.shapiro_wilk_normality,
                                "D'Agostino-Pearson": parametrics.d_agostino_pearson,
                                "Kolmogorov-Smirnov": parametrics.kolmogorov_smirnov,
                                }
    available_homocedasticity_test = {"Levene": parametrics.levene_test, 
                                      "Bartlett": parametrics.bartlett_test}
    content = []
    alpha = float(alpha)
    if not(test_normality is None):
        test_normality_function = available_normality_test[test_normality]
        columns = list(dataset.columns)
        statistic_list, p_value_list, cv_value_list, hypothesis_list = [], [], [], []

        for i in range(1, len(columns)):
            statistic, p_value, cv_value, hypothesis = test_normality_function(dataset[columns[i]].to_numpy(), alpha)
            # TODO realizar el bucle
            statistic_list.append(statistic)
            p_value_list.append(p_value)
            cv_value_list.append(cv_value)
            hypothesis_list.append(hypothesis)

        test_subtitle = html.H5(f"{test_normality} test (significance level of {alpha})")
        title = html.H3(f"Normality Analysis", className="title")

        results_test = pd.DataFrame({"Dataset": columns[1:], "Statistic": statistic_list, "p-value": p_value_list,
                                     "Results": hypothesis_list})

        table = generate_tabla_of_dataframe(results_test, height_table=f"{4.2 * len(results_test)}em")
        content = [title, test_subtitle, table]
    if not(test_homoscedasticity is None):
        test_homocedasticity_function = available_homocedasticity_test[test_homoscedasticity]
        statistic, p_value, cv_value, hypothesis = test_homocedasticity_function(dataset, alpha)
        if p_value is None:
            test_result = f"Statistic: {statistic} Critical Value: {cv_value}  Result: {hypothesis}"
        else:
            test_result = f"Statistic: {statistic} P-Value: {p_value}  Result: {hypothesis}"
        test_subtitle = html.H5(f"{test_homoscedasticity} test (significance level of {alpha})")
        title = html.H3(f"Homocedasticity Analysis", className="title")

        content.extend([title, test_subtitle, test_result])

    return content


@app.callback(Output('results_normality', 'children'),
              Output('results_normality', 'className'),
              Output("reset-norm", "n_clicks"),
              Input("submit-norm", "n_clicks"),
              Input("reset-norm", "n_clicks"),
              State('reports_switch', 'on'),
              State('selector_alpha', 'value'),
              State('selector_normality', 'value'),
              State('selector_homoscedasticity', 'value'),
              State('user-data', 'data'))
def process_normality(n_clicks, reset, generate_pdf, alpha, test_normality, test_homoscedasticity, current_data):
    if n_clicks is None and reset is None:
        return dash.no_update
    if reset:
        return html.Div(""), "", None

    dataset = pd.DataFrame(current_data)
    content = results_normality(dataset, alpha, test_normality, test_homoscedasticity)

    return html.Div(content), "", None


if __name__ == '__main__':
    app.run(debug=True)
