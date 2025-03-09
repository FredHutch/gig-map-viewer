import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium", app_title="Pangenome Viewer ")


@app.cell
def _(mo):
    mo.md(r"""# Pangenome Viewer (gig-map)""")
    return


@app.cell
def _():
    # Load the marimo library in a dedicated cell for efficiency
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # If the script is running in WASM (instead of local development mode), load micropip
    import sys
    if "pyodide" in sys.modules:
        import micropip
        running_in_wasm = True
    else:
        micropip = None
        running_in_wasm = False
    return micropip, running_in_wasm, sys


@app.cell
async def _(micropip, mo, running_in_wasm):
    with mo.status.spinner("Loading dependencies"):
        # If we are running in WASM, some dependencies need to be set up appropriately.
        # This is really just aligning the needs of the app with the default library versions
        # that come when a marimo app loads in WASM.
        if running_in_wasm:
            print("Installing via micropip")
            # Downgrade plotly to avoid the use of narwhals
            await micropip.install("plotly<6.0.0")
            await micropip.install("ssl")
            micropip.uninstall("urllib3")
            micropip.uninstall("httpx")
            await micropip.install("httpx==0.26.0")
            await micropip.install("urllib3==2.3.0")
            await micropip.install("botocore==1.36.23")
            await micropip.install("jmespath==1.0.1")
            await micropip.install("s3transfer==0.11.3")
            await micropip.install("boto3==1.36.23")
            await micropip.install("aiobotocore==2.20.0")
            await micropip.install("cirro[pyodide]>=1.2.16")

        from io import StringIO, BytesIO
        from queue import Queue
        from time import sleep
        from typing import Dict, Optional
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np
        from functools import lru_cache
        import base64
        from urllib.parse import quote_plus
        from scipy import cluster, spatial
        from anndata import AnnData
        from copy import copy

        from cirro import DataPortalLogin, DataPortalDataset
        from cirro.services.file import FileService
        from cirro.sdk.file import DataPortalFile
        from cirro.config import list_tenants
        from scipy import cluster

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()
    return (
        AnnData,
        BytesIO,
        DataPortalDataset,
        DataPortalFile,
        DataPortalLogin,
        Dict,
        FileService,
        Optional,
        Queue,
        StringIO,
        base64,
        cluster,
        copy,
        go,
        list_tenants,
        lru_cache,
        make_subplots,
        np,
        pd,
        px,
        pyodide_patch_all,
        quote_plus,
        sleep,
        spatial,
    )


@app.cell
def _(DataPortalDataset):
    # Define the types of datasets which can be read in
    # This is used to filter the dataset selector, below
    def filter_datasets_pangenome(dataset: DataPortalDataset):
        return "hutch-gig-map-build-pangenome" in dataset.process_id and dataset.status == "COMPLETED"
    return (filter_datasets_pangenome,)


@app.cell
def _(mo):
    # Get and set the query parameters
    query_params = mo.query_params()
    return (query_params,)


@app.cell
def _(list_tenants):
    # Get the tenants (organizations) available in Cirro
    tenants_by_name = {i["displayName"]: i for i in list_tenants()}
    tenants_by_domain = {i["domain"]: i for i in list_tenants()}


    def domain_to_name(domain):
        return tenants_by_domain.get(domain, {}).get("displayName")


    def name_to_domain(name):
        return tenants_by_name.get(name, {}).get("domain")
    return (
        domain_to_name,
        name_to_domain,
        tenants_by_domain,
        tenants_by_name,
    )


@app.cell
def _(mo):
    mo.md(r"""## Load Data""")
    return


@app.cell
def _(mo):
    # Use a state element to manage the Cirro client object
    get_client, set_client = mo.state(None)
    return get_client, set_client


@app.cell
def _(domain_to_name, mo, query_params, tenants_by_name):
    # Let the user select which tenant to log in to (using displayName)
    domain_ui = mo.ui.dropdown(
        options=tenants_by_name,
        value=domain_to_name(query_params.get("domain")),
        on_change=lambda i: query_params.set("domain", i["domain"]),
        label="Load Data from Cirro",
    )
    domain_ui
    return (domain_ui,)


@app.cell
def _(DataPortalLogin, domain_ui, get_client, mo):
    # If the user is not yet logged in, and a domain is selected, then give the user instructions for logging in
    # The configuration of this cell and the two below it serve the function of:
    #   1. Showing the user the login instructions if they have selected a Cirro domain
    #   2. Removing the login instructions as soon as they have completed the login flow
    if get_client() is None and domain_ui.value is not None:
        with mo.status.spinner("Authenticating"):
            # Use device code authorization to log in to Cirro
            cirro_login = DataPortalLogin(base_url=domain_ui.value["domain"])
            cirro_login_ui = mo.md(cirro_login.auth_message_markdown)
    else:
        cirro_login = None
        cirro_login_ui = None

    mo.stop(cirro_login is None)
    cirro_login_ui
    return cirro_login, cirro_login_ui


@app.cell
def _(cirro_login, set_client):
    # Once the user logs in, set the state for the client object
    set_client(cirro_login.await_completion())
    return


@app.cell
def _(get_client, mo):
    # Get the Cirro client object (but only take action if the user selected Cirro as the input)
    client = get_client()
    mo.stop(client is None)
    return (client,)


@app.cell
def _(get_client, mo):
    mo.stop(get_client() is not None)
    mo.md("*_Log in to view data_*")
    return


@app.cell
def _():
    # Helper functions for dealing with lists of objects that may be accessed by id or name
    def id_to_name(obj_list: list, id: str) -> str:
        if obj_list is not None:
            return {i.id: i.name for i in obj_list}.get(id)


    def name_to_id(obj_list: list) -> dict:
        if obj_list is not None:
            return {i.name: i.id for i in obj_list}
        else:
            return {}
    return id_to_name, name_to_id


@app.cell
def _(client):
    # Set the list of projects available to the user
    projects = client.list_projects()
    projects.sort(key=lambda i: i.name)
    return (projects,)


@app.cell
def _(id_to_name, mo, name_to_id, projects, query_params):
    # Let the user select which project to get data from
    project_ui = mo.ui.dropdown(
        label="Select Project:",
        value=id_to_name(projects, query_params.get("project")),
        options=name_to_id(projects),
        on_change=lambda i: query_params.set("project", i)
    )
    project_ui
    return (project_ui,)


@app.cell
def _(client, filter_datasets_pangenome, mo, project_ui):
    # Stop if the user has not selected a project
    mo.stop(project_ui.value is None)

    # Get the list of pangenome datasets available to the user
    pangenome_datasets = [
        dataset
        for dataset in client.get_project_by_id(project_ui.value).list_datasets()
        if filter_datasets_pangenome(dataset)
    ]
    pangenome_datasets.sort(key=lambda ds: ds.created_at, reverse=True)
    return (pangenome_datasets,)


@app.cell
def _(id_to_name, mo, name_to_id, pangenome_datasets, query_params):
    # Let the user select which pangenome dataset to get data from
    pangenome_dataset_ui = mo.ui.dropdown(
        label="Select Pangenome:",
        value=id_to_name(pangenome_datasets, query_params.get("dataset")),
        options=name_to_id(pangenome_datasets),
        on_change=lambda i: query_params.set("dataset", i)
    )
    pangenome_dataset_ui
    return (pangenome_dataset_ui,)


@app.cell
def _(client, mo, pangenome_dataset_ui, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(pangenome_dataset_ui.value is None)

    # Get the selected dataset
    pangenome_dataset = (
        client
        .get_project_by_id(project_ui.value)
        .get_dataset_by_id(pangenome_dataset_ui.value)
    )
    return (pangenome_dataset,)


@app.cell
def _(
    AnnData,
    DataPortalDataset,
    Dict,
    List,
    cluster,
    mo,
    pangenome_dataset,
    pd,
    spatial,
):
    # Define an object with all of the information for a pangenome
    class Pangenome:
        """
        Attributes:
            ds: DataPortalDataset
            bin_contents: Dict[str, pd.DataFrame]
            adata: AnnData
                obs: genomes
                var: bins
        """

        ds: DataPortalDataset
        adata: AnnData
        bin_contents: Dict[str, pd.DataFrame]
        genome_tree: cluster.hierarchy.ClusterNode
        tree_df: pd.DataFrame
        clades: List[set]

        def __init__(self, ds: DataPortalDataset):
            self.ds = ds

            # Read in the table of which genes are in which bins
            # and turn into a dict keyed by bin ID
            self.bin_contents = {
                bin: d.drop(columns=['bin'])
                for bin, d in self.read_csv("data/bin_pangenome/gene_bins.csv").groupby("bin")
            }

            # Build an AnnData object
            self.adata = self._make_adata()

            # Run linkage clustering on the genomes
            self._genome_tree, self._genome_tree_nodes = cluster.hierarchy.to_tree(
                cluster.hierarchy.linkage(
                    spatial.distance.squareform(
                        self.adata.obsp["ani_distance"]
                    ),
                    method="average"
                ),
                rd=True
            )

            # Get all of the possible monophyletic groupings of genomes, based on that binary tree
            self.index_tree()

        def index_tree(self):
            """Get all of the possible monophyletic groupings of genomes, based on that binary tree."""

            self.clades = [
                set([
                    self.adata.obs_names[ix]
                    for ix in cn.pre_order(lambda n: n.id)
                    if ix < self.adata.obs.shape[0]
                ])
                for cn in self._genome_tree_nodes
                if cn.get_count() > 1
            ]

        def _make_adata(self) -> AnnData:

            # The variable annotations includes the number of genes for each variable
            var = pd.DataFrame(dict(
                n_genes=pd.Series({bin: d.shape[0] for bin, d in self.bin_contents.items()})
            ))

            # Read in the table listing which genomes contain which bins
            _genome_contents = self.read_csv("data/bin_pangenome/genome_content.long.csv")

            # Split off the genome annotation information into a separate table
            obs = (
                _genome_contents
                .drop(columns=["bin", "n_genes_detected", "prop_genes_detected"])
                .drop_duplicates()
                .set_index("genome")
            )

            # X will be the proportion of genes detected
            X = (
                _genome_contents
                .pivot(
                    index="genome",
                    columns="bin",
                    values="prop_genes_detected"
                )
                .fillna(0)
            )

            # Read in the ANI distance matrix for all genomes
            _genome_ani = self.read_csv(
                "data/distances.csv.gz",
                index_col=0
            )

            # Build an AnnData object
            return AnnData(
                X=X,
                obs=obs.reindex(index=X.index),
                var=var.reindex(index=X.columns),
                obsp=dict(
                    ani_distance=_genome_ani.reindex(
                        columns=X.index,
                        index=X.index
                    )
                )
            )

        def read_csv(self, fp: str, **kwargs):
            return self.ds.list_files().get_by_id(fp).read_csv(**kwargs)

    with mo.status.spinner("Loading data..."):
        pg = Pangenome(pangenome_dataset)
    return Pangenome, pg


@app.cell
def _(mo, pangenome_dataset_ui):
    # The user can apply filters for visualizing the pangenome
    pangenome_args = (
        mo.md("""
    ### Pangenome Options

    - {min_prop}
    - {min_genome_size}
        """)
        .batch(
            min_prop=mo.ui.number(
                label="Minimum Proportion of Genes in Bin:",
                start=0.01,
                stop=1.0,
                value=0.9
            ),
            min_genome_size=mo.ui.number(
                label="Minimum Genome Size (# of Genes):",
                start=1,
                value=800
            )
        )
    )
    # Stop if the user has not selected a dataset
    mo.stop(pangenome_dataset_ui.value is None)
    pangenome_args
    return (pangenome_args,)


@app.cell
def _(
    AnnData,
    Pangenome,
    cluster,
    copy,
    make_subplots,
    mo,
    np,
    pangenome_args,
    pd,
    pg,
    px,
):
    # Based on the inputs, format and present a display
    class PangenomeDisplay:
        pg: Pangenome
        # Filtered and annotated copy of the pg.adata
        adata: AnnData

        def __init__(self, pg: Pangenome, min_prop: float, min_genome_size: int):
            self.pg = pg

            # Copy the AnnData object
            self.adata = copy(pg.adata)

            # Make a binary mask layer indicating whether each genome meets the min_prop threshold
            self.adata.layers["present"] = (self.adata.to_df() >= min_prop).astype(int)

            # Calculate the number of genes detected per genome
            # (counting each entire bin that was detected per genome)
            self.adata.obs["n_genes"] = pd.Series({
                genome: self.adata.var.loc[genome_contains_bin, "n_genes"].sum()
                for genome, genome_contains_bin in (self.adata.to_df(layer="present") == 1).iterrows()
            })

            # Filter the genomes based on the size filter
            self.adata = self.adata[self.adata.obs["n_genes"] >= min_genome_size, :]

            # Compute the number of genomes that each bin is found in
            self.adata.var["n_genomes"] = self.adata.to_df(layer="present").sum()

            # Filter out any bins which are found in 0 genomes
            self.adata = self.adata[:, self.adata.var["n_genomes"] > 0]

            # Compute the proportion of genomes that each bin is found in
            self.adata.var["prevalence"] = self.adata.var["n_genomes"] / self.adata.shape[0]

            # Compute the monophyly score for each bin, and then summarize across each genome
            self.compute_monophyly()

        def compute_monophyly(self):
            """
            Compute the monophyly score for each bin.
            (# of genomes the bin is present in) / 
            (# of genomes in the smallest monophyletic clade containing those genomes)
            """

            # Update the clades to only include what is present in the filtered dataset
            self.pg.clades = [
                clade & set(self.adata.obs_names.values)
                for clade in self.pg.clades
            ]

            # Annotate each bin by its monophyly score
            self.adata.var["monophyly"] = pd.Series({
                bin: self._compute_monophyly(bin)
                for bin in self.adata.var_names
            })

            # Annotate each genome by the weighted average of the monophyly score for each
            # bin it contains (weighted by the bin size)
            self.adata.obs["monophyly"] = pd.Series({
                genome: (
                    self.adata.var
                    .loc[genome_contains_bin]
                    .apply(
                        lambda r: r['monophyly'] * r['n_genes'],
                        axis=1
                    )
                    .sum()
                    / self.adata.obs.loc[genome, "n_genes"]
                )
                for genome, genome_contains_bin in (self.adata.to_df(layer="present") == 1).iterrows()
            })

        def _compute_monophyly(self, bin: str):

            # Get the set of genomes which contain this bin
            genomes = set(self.adata.obs_names.values[self.adata.to_df(layer="present")[bin] == 1])
            assert len(genomes) == self.adata.to_df(layer="present")[bin].sum(), bin

            # If there are none, stop
            if len(genomes) == 0:
                return

            # If there is only one, return 1
            elif len(genomes) == 1:
                return 1

            # If there are more, start by finding the smallest clade which contains all of the genomes
            else:
                smallest_clade = np.min([
                    len(clade)
                    for clade in self.pg.clades
                    if genomes <= clade and len(clade) > 0
                ])
                # assert len(genomes) < 3, (genomes, smallest_clade)
                # Return the proportion of genomes in that clade which contain the bin
                return len(genomes) / smallest_clade

        # def plot_bins_scatter(self):
        #     """Make a scatter plot with bin information."""

        def plot_heatmap(self, min_bin_size=1, height=500):
            """Make a heatmap showing which bins are present in which genomes."""

            present = (
                self.adata
                [:, self.adata.var["n_genes"] >= min_bin_size]
                .to_df(layer="present")
            )

            genome_order = sort_axis(present)
            bin_order = sort_axis(present.T)

            fig = make_subplots(
                shared_xaxes=True,
                shared_yaxes=True,
                rows=4,
                cols=3,
                start_cell="bottom-left",
                column_widths=[2, 1, 1],
                row_heights=[2, 1, 1, 1],
                horizontal_spacing=0.01,
                vertical_spacing=0.01
            )
            fig.add_heatmap(
                x=bin_order,
                y=genome_order,
                z=present.reindex(index=genome_order, columns=bin_order),
                text=[
                    [
                        f"{genome}<br>{bin}<br>{'Present' if present.loc[genome, bin] else 'Absent'}<extra></extra>"
                        for bin in bin_order
                    ]
                    for genome in genome_order
                ],
                hovertemplate="%{text}<extra></extra>",
                colorscale="Blues",
                row=1,
                col=1,
                showscale=False,
            )
            fig.add_bar(
                y=self.adata.var["n_genes"].reindex(bin_order),
                x=bin_order,
                row=2,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.adata.var.loc[bin_id, 'n_genes']:,} Genes"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                x=self.adata.obs["n_genes"].reindex(genome_order),
                y=genome_order,
                orientation='h',
                row=1,
                col=2,
                showlegend=False,
                text=[
                    f"{genome_id}<br>{self.adata.obs.loc[genome_id, 'n_genes']:,} Genes"
                    for genome_id in genome_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                x=self.adata.obs["monophyly"].reindex(genome_order),
                y=genome_order,
                orientation='h',
                row=1,
                col=3,
                showlegend=False,
                text=[
                    f"{genome_id}<br>{self.adata.obs.loc[genome_id, 'monophyly']:,} Monophyly Score"
                    for genome_id in genome_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                y=self.adata.var["monophyly"].reindex(bin_order),
                x=bin_order,
                row=3,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.adata.var.loc[bin_id, 'monophyly']:,} Monophyly Score"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                y=self.adata.var["prevalence"].reindex(bin_order),
                x=bin_order,
                row=4,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.adata.var.loc[bin_id, 'prevalence']:,} Prevalence"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.update_layout(
                height=height,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                bargap=0,
                yaxis4=dict(
                    title_text="Bin Size<br>(# of Genes)",
                    type="log",
                ),
                yaxis7=dict(title_text="Monophyly<br>Score"),
                yaxis10=dict(title_text="Prevalence"),
                yaxis=dict(
                    # side="right",
                    title_text="Genome<br>Presence / Absence",
                    showticklabels=False
                ),
                xaxis2=dict(
                    title_text="Genome Size (# of Genes)"
                ),
                xaxis3=dict(
                    title_text="Monophyly Score"
                )
            )
            return fig

        def plot_scatter(self):
            """Show summary metrics on bins and genomes."""

            fig = px.scatter(
                self.adata.var.assign(log_n_genes=self.adata.var['n_genes'].apply(np.log10)).reset_index(),
                hover_name="bin",
                x="prevalence",
                color="monophyly",
                y="n_genes",
                size="log_n_genes",
                template="simple_white",
                color_continuous_scale="bluered",
                log_x=True,
                log_y=True,
                labels=dict(
                    n_genes="Bin Size (# of Genes)",
                    log_n_genes="Log Bin Size (log10 - # of Genes)",
                    prevalence="Bin Prevalence (Proportion of Genomes)",
                    monophyly="Monophyly Score"
                )
            )
            return fig


    def sort_axis(df: pd.DataFrame):
        return df.index.values[
            cluster.hierarchy.leaves_list(
                cluster.hierarchy.linkage(
                    df.values,
                    method="average",
                    metric="jaccard"
                )
            )
        ]

    with mo.status.spinner("Computing layout..."):
        pg_display = PangenomeDisplay(pg, **pangenome_args.value)
    return PangenomeDisplay, pg_display, sort_axis


@app.cell
def _():
    # pg_display.adata.var
    return


@app.cell
def _():
    # pg_display.adata.obs
    return


@app.cell
def _(mo, pangenome_dataset_ui):
    heatmap_options_ui = mo.md("""
    ### Genome Heatmap Options

     - {min_bin_size}
     - {height}
    """).batch(
            min_bin_size=mo.ui.number(
                label="Minimum Bin Size (# of Genes):",
                start=1,
                value=100
            ),
            height=mo.ui.number(
                label="Figure Height",
                start=100,
                value=500
            )
    )
    # Stop if the user has not selected a dataset
    mo.stop(pangenome_dataset_ui.value is None)
    heatmap_options_ui
    return (heatmap_options_ui,)


@app.cell
def _(heatmap_options_ui, pg_display):
    pg_display.plot_heatmap(**heatmap_options_ui.value)
    return


@app.cell
def _(pg_display):
    pg_display.plot_scatter()
    return


@app.cell
def _(mo, pg, pg_display):
    # Show the user information about a specific bin
    display_bin_args = (
        mo.md(
            """
    ### Show Bin Information

    {bin_id}

    {genome_columns}
            """
        )
        .batch(
            bin_id=mo.ui.dropdown(
                label="Display:",
                options=sorted(pg.adata.var_names, key=lambda bin_id: int(bin_id.split(" ")[-1])),
                value="Bin 1"
            ),
            genome_columns=mo.ui.multiselect(
                label="Show Columns:",
                options=pg_display.adata.obs.columns.values,
                value=pg_display.adata.obs.columns.values
            )
        )
    )
    display_bin_args
    return (display_bin_args,)


@app.cell
def _(mo, pg, pg_display):
    def display_bin(bin_id, genome_columns):
        # Show the list of genes in the bin
        gene_df = (
            pg.bin_contents[bin_id]
            .set_index("gene_id")
            .rename(
                columns=dict(
                    combined_name="Gene Annotation",
                    n_genomes="Number of Genomes"
                )
            )
        )

        # Show the list of genomes containing the bin
        genome_df = (
            pg_display.adata
            [
                pg_display.adata
                .to_df(layer="present")
                [bin_id] == 1
            ]
            .obs
        )

        return mo.vstack([
            mo.md(bin_id),
            mo.md(f"{gene_df.shape[0]:,} Genes / {genome_df.shape[0]:,} Genomes"),
            gene_df,
            genome_df.reindex(columns=genome_columns)
        ])
    return (display_bin,)


@app.cell
def _(display_bin, display_bin_args):
    display_bin(**display_bin_args.value)
    return


@app.cell
def _(mo, pg):
    # Show the user overlap between bins
    bin_overlap_args = mo.md("""
    ### Show Bin Overlap

    {bins}

    {height}

    """).batch(
        bins=mo.ui.multiselect(
            label="Bins:",
            options=sorted(pg.adata.var_names, key=lambda bin_id: int(bin_id.split(" ")[-1])),
            value=["Bin 1", "Bin 2"]
        ),
        height=mo.ui.number(
            label="Figure Height:",
            start=100,
            value=500
        )
    )
    bin_overlap_args
    return (bin_overlap_args,)


@app.cell
def _(mo, pangenome_dataset_ui):
    mo.stop(pangenome_dataset_ui.value is None)
    genome_annot_enabled_ui=mo.ui.checkbox(
        label="Annotate Genomes",
        value=False
    )
    genome_annot_enabled_ui
    return (genome_annot_enabled_ui,)


@app.cell
def _(genome_annot_enabled_ui, mo, pg):
    genome_annot_ui=mo.ui.dropdown(
        label="Annotation Column:",
        options=(
            ['None'] + pg.adata.obs_keys()
            if genome_annot_enabled_ui.value
            else ['None']
        )
        ,
        value=(
            "assemblyInfo_biosample_description_organism_organismName"
            if ("assemblyInfo_biosample_description_organism_organismName" in pg.adata.obs_keys()) and genome_annot_enabled_ui.value
            else 'None'
        )
    )
    genome_annot_ui
    return (genome_annot_ui,)


@app.cell
def _(genome_annot_ui, mo, pg):
    # If an annotation is selected, let the user pick the groups
    def genome_annot_groups_options():
        return (
            []
            if genome_annot_ui.value == 'None'
            else
            pg.adata.obs[genome_annot_ui.value].drop_duplicates().sort_values().values
        )

    genome_annot_groups = mo.ui.multiselect(
        label="Include Groups:",
        options=genome_annot_groups_options(),
        value=genome_annot_groups_options()
    )
    genome_annot_groups
    return genome_annot_groups, genome_annot_groups_options


@app.cell
def _(make_subplots, pd, pg_display):
    def show_bin_overlap(bins: list, height: int, genome_annot: str, genome_annot_groups=[]):
        # Get the table of which genomes have these bins
        presence = (
            pg_display
            .adata
            .to_df(layer="present")
            .reindex(columns=bins)
            .groupby(bins)
            .apply(len, include_groups=False)
            .sort_values(ascending=False)
            .reset_index()
        )

        # Make a plot with two panels
        fig = make_subplots(
            shared_xaxes=True,
            rows=2 + int(genome_annot != 'None'),
            cols=1,
            start_cell="bottom-left",
            # column_widths=[2, 1, 1],
            # row_heights=[2, 1, 1, 1],
            horizontal_spacing=0.01,
            vertical_spacing=0.1
        )

        # Make a long form table for the points
        points = pd.DataFrame([
            dict(
                x=row_i,
                y=bins.index(bin),
                color="black" if present else "white",
                bin=bin,
                text=f"{bin} is {'Present' if present else 'Absent'}",
                present=present
            )
            for row_i, row in presence.reindex(columns=bins).iterrows()
            for bin, present in row.items()
        ])

        # Add the vertical lines
        for x, x_df in points.query('present == 1').groupby("x"):
            if x_df.shape[0] > 1:
                fig.add_scatter(
                    x=[x, x],
                    y=[x_df['y'].min(), x_df['y'].max()],
                    mode="lines",
                    line=dict(
                        color="grey"
                    ),
                    showlegend=False
                )

        fig.add_scatter(
            x=points["x"],
            y=points["y"],
            mode="markers",
            marker=dict(
                color=points["color"],
                size=10,
                line=dict(
                    width=1,
                    color="black"
                )
            ),
            text=points["text"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        )

        fig.add_bar(
            x=list(range(presence.shape[0])),
            y=presence[0],
            row=2,
            col=1,
            showlegend=False,
            hovertemplate="%{y:,} Genomes<extra></extra>",
            marker=dict(color="black")
        )
        # Optionally add the annotation table
        if genome_annot != 'None':
            annot_df = (
                pd.concat([
                    pg_display
                    .adata
                    .to_df(layer="present")
                    .reindex(columns=bins),
                    pg_display
                    .adata
                    .obs
                    .reindex(columns=[genome_annot])
                ], axis=1)
                .groupby(bins + [genome_annot])
                .apply(len, include_groups=False)
                .sort_values(ascending=False)
                .reset_index()
                .pivot_table(
                    columns=bins,
                    index=genome_annot,
                    values=0
                )
                .reindex(
                    columns=presence.set_index(bins).index
                )
                .fillna(0)
            )
            if len(genome_annot_groups) > 0:
                annot_df = annot_df.reindex(index=genome_annot_groups)
            fig.add_heatmap(
                z=annot_df.values,
                y=annot_df.index.values,
                row=3,
                col=1,
                colorscale="blues",
                hovertemplate="%{y}<br>%{z} Genomes<extra></extra>"
            )
        fig.update_layout(
            template="simple_white",
            xaxis=dict(
                showticklabels=False,
                title_text="Presence / Absence"
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(bins))),
                ticktext=bins
            ),
            yaxis2=dict(
                title_text="Number of Genomes"
            ),
            height=height
        )
        return fig

        return presence
    return (show_bin_overlap,)


@app.cell
def _(
    bin_overlap_args,
    genome_annot_groups,
    genome_annot_ui,
    show_bin_overlap,
):
    show_bin_overlap(
        genome_annot_groups=genome_annot_groups.value,
        genome_annot=genome_annot_ui.value,
        **bin_overlap_args.value
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
