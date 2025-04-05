import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium", app_title="Pangenome Viewer ")


@app.cell
def _(mo):
    mo.sidebar(
        [
            mo.nav_menu(
                {
                    "#pangenome-viewer-gig-map": f"{mo.icon('lucide:info')} About",
                    "#connect-to-database": f"{mo.icon('lucide:database')} Connect",
                    "#inspect-pangenome": f"{mo.icon('lucide:map')} Inspect Pangenome",
                    "#compare-pangenomes": f"{mo.icon('lucide:map-plus')} Compare Pangenomes",
                    "#inspect-gene-bin": f"{mo.icon('lucide:box')} Inspect Gene Bin",
                    "#compare-gene-bins": f"{mo.icon('lucide:boxes')} Compare Gene Bins",
                    "#analyze-metagenomes": f"{mo.icon('lucide:test-tubes')} Analyze Metagenomes",
                    "#settings": f"{mo.icon('lucide:settings')} Settings"
                },
                orientation="vertical",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Pangenome Viewer (gig-map)

        The [gig-map (for "genes-in-genomes map")](https://github.com/FredHutch/gig-map/wiki) tool
        is used to analyze collections of genomes ("pan-genomes") according to which genes are shared
        among different groups of genomes.
        Genes are organized into **bins** -- groups of genes which are all generally found in the same genomes.

        Each bin represents a group of genes which is either (a) preserved together within a lineage, or (b)
        consistently transferred together between lineages.
        Biologically, these tend to be species/genus core genomes, operons, genomic islands, plasmids, bacteriophages, etc.
        """
    )
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
            await micropip.install("cirro[pyodide]==1.2.16")

        from io import StringIO, BytesIO
        from queue import Queue
        from time import sleep
        from typing import Dict, Optional, List
        from functools import lru_cache
        import base64
        from urllib.parse import quote_plus
        from copy import copy

        from cirro import DataPortalLogin, DataPortalDataset
        from cirro.services.file import FileService
        from cirro.sdk.file import DataPortalFile
        from cirro.config import list_tenants

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()
    return (
        BytesIO,
        DataPortalDataset,
        DataPortalFile,
        DataPortalLogin,
        Dict,
        FileService,
        List,
        Optional,
        Queue,
        StringIO,
        base64,
        copy,
        list_tenants,
        lru_cache,
        pyodide_patch_all,
        quote_plus,
        sleep,
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
    return domain_to_name, name_to_domain, tenants_by_domain, tenants_by_name


@app.cell
def _(mo):
    mo.md(r"""## Connect to Database""")
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
def _(mo):
    mo.md(r"""## Inspect Pangenome""")
    return


@app.cell
def _(id_to_name, mo, name_to_id, pangenome_datasets, query_params):
    # Let the user select which pangenome dataset to get data from
    pangenome_dataset_ui = mo.ui.dropdown(
        label="Select Pangenome:",
        value=id_to_name(pangenome_datasets, query_params.get("inspect_pangenome")),
        options=name_to_id(pangenome_datasets),
        on_change=lambda i: query_params.set("inspect_pangenome", i)
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
def _(mo, pangenome_dataset):
    mo.accordion({
        "Analysis Settings": mo.md("""
    | Analysis Settings | |
    | --- | --- |
    | Gene Clustering Similarity | {gene_catalog_cluster_similarity}% |
    | Gene Clustering Overlap | {gene_catalog_cluster_similarity}% |
    | Minimum Gene Length | {gene_catalog_min_gene_length:,}aa |
    | Alignment Minimum Coverage | {alignment_min_coverage}% |
    | Alignment Minimum Identity | {alignment_min_identity}% |
    | Gene Binning Threshold | {clustering_max_dist_genes}% |
    | Minimum Gene Bin Size | {clustering_min_bin_size} |
    | Minimum Genomes Per Gene | {clustering_min_genomes_per_gene} |

        """.format(
            **{
                f"{group}_{kw}": (
                    int(val * 100) if kw in ["cluster_similarity", "cluster_coverage", "max_dist_genes"]
                    else val
                )
                for group, group_attrs in pangenome_dataset.params.additional_properties.items()
                for kw, val in group_attrs.items()
            }
        ))
    })
    return


@app.cell
def _(mo):
    with mo.status.spinner("Loading dependencies"):
        import pandas as pd
        from scipy import cluster, spatial
        from anndata import AnnData
        import sklearn
    return AnnData, cluster, pd, sklearn, spatial


@app.cell
def _(
    AnnData,
    DataPortalDataset,
    Dict,
    List,
    cluster,
    lru_cache,
    mo,
    np,
    pd,
    query_params,
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

        def __init__(self, ds: DataPortalDataset, min_prop: float):
            self.ds = ds

            # Read in the table of which genes are in which bins
            # and turn into a dict keyed by bin ID
            self.bin_contents = {
                bin: d.drop(columns=['bin'])
                for bin, d in self.read_csv(
                    "data/bin_pangenome/gene_bins.csv",
                    low_memory=False
                ).groupby("bin")
            }

            # Build an AnnData object
            self.adata = self._make_adata()

            # Make a binary mask layer indicating whether each genome meets the min_prop threshold
            self.adata.layers["present"] = (self.adata.to_df() >= min_prop).astype(int)

            # Calculate the number of genes detected per genome
            # (counting each entire bin that was detected per genome)
            self.adata.obs["n_genes"] = pd.Series({
                genome: self.adata.var.loc[genome_contains_bin, "n_genes"].sum()
                for genome, genome_contains_bin in (self.adata.to_df(layer="present") == 1).iterrows()
            })

            # Filter out the genomes which are below the threshold
            _n_genomes_filter_threshold = (
                self.adata.obs['n_genes'].median()
                *
                float(query_params.get("min_genes_rel_median", '0.5'))
            )
            _n_genomes_filter = (self.adata.obs['n_genes'] >= _n_genomes_filter_threshold)
            self.n_genomes_filtered_out = self.adata.shape[0] - _n_genomes_filter.sum()
            self.adata = self.adata[_n_genomes_filter]

            # Filter the dataset to only include genomes which include a gene
            self.adata = self.adata[self.adata.obs["n_genes"] > 0]

            # Compute the number of genomes that each bin is found in
            self.adata.var["n_genomes"] = self.adata.to_df(layer="present").sum()

            # Filter out any bins which are found in 0 genomes
            self.adata = self.adata[:, self.adata.var["n_genomes"] > 0]

            # Compute the proportion of genomes that each bin is found in
            self.adata.var["prevalence"] = self.adata.var["n_genomes"] / self.adata.shape[0]

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

            # Compute the monophyly score for each bin, and then summarize across each genome
            self.compute_monophyly()

        def compute_monophyly(self):
            """
            Compute the monophyly score for each bin.
            (# of genomes the bin is present in) / 
            (# of genomes in the smallest monophyletic clade containing those genomes)
            """

            # Update the clades to only include what is present in the filtered dataset
            self.clades = [
                clade & set(self.adata.obs_names.values)
                for clade in self.clades
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
                    for clade in self.clades
                    if genomes <= clade and len(clade) > 0
                ])
                # assert len(genomes) < 3, (genomes, smallest_clade)
                # Return the proportion of genomes in that clade which contain the bin
                return len(genomes) / smallest_clade

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
            _genome_contents = self.read_csv(
                "data/bin_pangenome/genome_content.long.csv",
                low_memory=False
            )

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


    # Cache the creation of pangenome objects
    @lru_cache
    def make_pangenome(ds: DataPortalDataset, min_prop: float):
        with mo.status.spinner("Loading data..."):
            return Pangenome(ds, min_prop)
    return Pangenome, make_pangenome


@app.cell(hide_code=True)
def _(mo):
    with mo.status.spinner("Loading dependencies:"):
        import plotly.express as px
        from plotly.subplots import make_subplots
        import numpy as np
    return make_subplots, np, px


@app.cell
def _(
    DataPortalDataset,
    Pangenome,
    cluster,
    make_pangenome,
    make_subplots,
    mo,
    np,
    pd,
    px,
    query_params,
    sklearn,
):
    # Object used to configure and display the "Inspect Pangenome" section
    class InspectPangenome:
        pg: Pangenome

        def __init__(self, ds: DataPortalDataset):
            self.pg = make_pangenome(ds, float(query_params.get("min_prop", 0.5)))

            # Precompute the coordinates for the genomes
            # Ordinate using the log-transformed gene abundances, to not overrepresent large bins
            log_genes_per_genome = self.pg.adata.to_df(layer="present") * self.pg.adata.var["n_genes"].apply(np.log10)
            self.genome_coords = sklearn.manifold.Isomap().fit_transform(log_genes_per_genome)

            # Make a table with the number of genes from each bin which are found in each genome
            self.n_genes_per_genome = self.pg.adata.to_df(layer="present") * self.pg.adata.var["n_genes"]
    
            # Calculate the proportion of each genome made up by each bin
            self.prop_genes_per_genome = self.n_genes_per_genome.apply(lambda r: r / r.sum(), axis=1)

        def plot_heatmap_args(self):
            return mo.md("""
    ### Inspect Pangenome: Heatmap Options

     - {top_n_bins}
     - {include_bins}
     - {height}
            """).batch(
                top_n_bins=mo.ui.number(
                    label="Show N Bins (Largest # of Genes):",
                    start=0,
                    value=40,
                    step=1
                ),
                include_bins=mo.ui.multiselect(
                    label="Show Specific Bins:",
                    value=[],
                    options=self.pg.adata.var_names
                ),
                height=mo.ui.number(
                    label="Figure Height",
                    start=100,
                    value=800
                )
            )

        def plot_heatmap(self, top_n_bins: int, height: int, include_bins: list):
            """Make a heatmap showing which bins are present in which genomes."""

            # First get the top bins selected based on bin size
            if top_n_bins is not None and top_n_bins > 0:
                bins_to_plot = list(self.pg.adata.var.head(top_n_bins).index.values)
            else:
                bins_to_plot = []

            for bin in include_bins:
                if bin not in bins_to_plot:
                    bins_to_plot.append(bin)

            if len(bins_to_plot) < 2:
                return mo.md("""Please select multiple bins to plot""")

            present = (
                self.pg.adata
                [:, bins_to_plot]
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
                y=self.pg.adata.var["n_genes"].reindex(bin_order),
                x=bin_order,
                row=2,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.pg.adata.var.loc[bin_id, 'n_genes']:,} Genes"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                x=self.pg.adata.obs["n_genes"].reindex(genome_order),
                y=genome_order,
                orientation='h',
                row=1,
                col=2,
                showlegend=False,
                text=[
                    f"{genome_id}<br>{self.pg.adata.obs.loc[genome_id, 'n_genes']:,} Genes"
                    for genome_id in genome_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                x=self.pg.adata.obs["monophyly"].reindex(genome_order),
                y=genome_order,
                orientation='h',
                row=1,
                col=3,
                showlegend=False,
                text=[
                    f"{genome_id}<br>{self.pg.adata.obs.loc[genome_id, 'monophyly']:,} Monophyly Score"
                    for genome_id in genome_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                y=self.pg.adata.var["monophyly"].reindex(bin_order),
                x=bin_order,
                row=3,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.pg.adata.var.loc[bin_id, 'monophyly']:,} Monophyly Score"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                y=self.pg.adata.var["prevalence"].reindex(bin_order),
                x=bin_order,
                row=4,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.pg.adata.var.loc[bin_id, 'prevalence']:,} Prevalence"
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

        def summary_metrics(self):
            return mo.accordion({
                "Pangenome Characteristics": mo.md(f"""
                | Pangenome Characteristics | |
                | --- | --- |
                | Number of Genomes | {self.pg.adata.n_obs:,} |
                | Number of Genes | {self.pg.adata.var["n_genes"].sum():,} |
                | Number of Gene Bins | {self.pg.adata.n_vars:,} |
                | Genome Size (max) | {self.pg.adata.obs["n_genes"].max()} |
                | Genome Size (75%) | {int(self.pg.adata.obs["n_genes"].quantile(0.75))} |
                | Genome Size (50%) | {int(self.pg.adata.obs["n_genes"].quantile(0.5))} |
                | Genome Size (25%) | {int(self.pg.adata.obs["n_genes"].quantile(0.25))} |
                | Genome Size (min) | {self.pg.adata.obs["n_genes"].min()} |
                | Genomes Omitted: Low # of Genes | {self.pg.n_genomes_filtered_out:,} |
                """)
            })

        def plot_scatter_args(self):
            return mo.md("""
    ### Inspect Pangenome: Genome Map

    - Show N Bins: {n_bins}
    - Annotate Genomes By: {annotate_by}
    - Annotation Offset (x): {label_offset_x}
    - Annotation Offset (y): {label_offset_y}
    - Omit Genomes: {omit_genomes}
    - Include Specific Bins: {include_bins}
    - {height}
            """).batch(
                annotate_by=mo.ui.dropdown(
                    options=self.pg.adata.obs.reset_index().columns.values,
                    value=query_params.get("inspect_genomes_annotate_by", self.pg.adata.obs.index.name),
                    on_change=lambda val: query_params.set("inspect_genomes_annotate_by", val)
                ),
                label_offset_x=mo.ui.number(value=0.),
                label_offset_y=mo.ui.number(value=0.5),
                n_bins=mo.ui.number(
                    start=0,
                    stop=100,
                    step=1,
                    value=10
                ),
                omit_genomes=mo.ui.multiselect(
                    options=self.pg.adata.obs_names,
                    value=[]
                ),
                include_bins=mo.ui.multiselect(
                    options=self.pg.adata.var_names,
                    value=[]
                ),
                height=mo.ui.number(
                    label="Figure Height",
                    start=100,
                    value=600
                )
            )

        def plot_scatter(
            self,
            annotate_by: str,
            n_bins: int,
            omit_genomes: list,
            include_bins: list,
            height: int,
            label_offset_x: float,
            label_offset_y: float
        ):
            """
            Show a scatter plot where each point is a genome.
            The layout is driven by the similarity of which genes are encoded in the genome.
            Annotations can be made according to the presence of different bins, genome size, etc.
            """
            if n_bins == 0 and len(include_bins) == 0:
                return mo.md("Select bins to display")

            with mo.status.spinner("Generating Plot:"):
        
                # Choose the bins to plot based on the mean proportion across genomes
                bins_to_plot = list(self.prop_genes_per_genome.mean().sort_values().tail(n_bins).index.values)

                # Add any specific bins selected by the user
                for _bin in include_bins:
                    if _bin not in bins_to_plot:
                        bins_to_plot.append(_bin)
        
                # Drop all of the unselected bins, and add an "Other" column
                n_genes_per_genome = pd.concat([
                    self.n_genes_per_genome.reindex(columns=bins_to_plot),
                    pd.DataFrame(dict(Other=self.n_genes_per_genome.drop(columns=bins_to_plot).sum(axis=1)))
                ], axis=1)
        
                # Make a label for each genome
                genome_label = {
                    genome: "<br>".join([
                        f"{bin}:&#9;{n_genes:,} genes"
                        for bin, n_genes
                        in genome_count.sort_values(ascending=False).items()
                        if n_genes > 0
                    ])
                    for genome, genome_count in n_genes_per_genome.iterrows()
                }
        
                # Make a table to use only for point size
                # For each genome, take the cumulative sum from smallest to largest
                point_size_per_genome = (
                    n_genes_per_genome
                    .apply(lambda r: (r > 0).apply(int), axis=1)
                    .apply(lambda r: r.iloc[::-1].cumsum(), axis=1)
                    .apply(lambda r: r / r.max(), axis=1)
                )
        
                # Make a table to plot which includes the proportion for every single bin
                plot_df = pd.DataFrame([
                    dict(
                        genome=genome,
                        x=genome_coord[0],
                        y=genome_coord[1],
                        bin=bin,
                        genes_in_bin=n_genes_per_genome.loc[genome, bin],
                        point_size=point_size_per_genome.loc[genome, bin],
                        text=genome_label[genome]
                    )
                    for genome, genome_coord in zip(self.pg.adata.obs_names, self.genome_coords)
                    for bin in point_size_per_genome.columns.values
                    if n_genes_per_genome.loc[genome, bin] > 0 and genome not in omit_genomes
                ]).sort_values(by="point_size", ascending=False)
        
                # Add the genome metadata
                plot_df = plot_df.merge(
                    self.pg.adata.obs,
                    right_index=True,
                    left_on="genome"
                )
        
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    custom_data=[annotate_by, "text"],
                    hover_data=["genes_in_bin"],
                    size="point_size",
                    color="bin",
                    hover_name=annotate_by,
                    template="simple_white",
                    opacity=1,
                    labels=dict(bin="Gene Bin")
                )
                fig.add_scatter(
                    x=self.genome_coords[:, 0] + label_offset_x,
                    y=self.genome_coords[:, 1] + label_offset_y,
                    text=[
                        (
                            genome if annotate_by == "genome"
                            else
                            self.pg.adata.obs.loc[genome, annotate_by]
                        )
                        for genome in self.pg.adata.obs_names
                    ],
                    mode="text",
                    legendgroup="Annotation",
                    legendgrouptitle=dict(text="Annotation"),
                    name=annotate_by
                )
        
                fig.update_traces(
                    hovertemplate="<b>%{customdata[0]}</b><br><br>%{customdata[1]}<extra></extra>",
                    marker=dict(line=dict(width=0))
                )
                fig.update_layout(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=height
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
    return InspectPangenome, sort_axis


@app.cell
def _(InspectPangenome, pangenome_dataset):
    # Initialize the object using the dataset selected by the user
    inspect_pangenome = InspectPangenome(pangenome_dataset)
    return (inspect_pangenome,)


@app.cell
def _(inspect_pangenome):
    inspect_pangenome.summary_metrics()
    return


@app.cell
def _(inspect_pangenome):
    inspect_pangenome_heatmap_args = inspect_pangenome.plot_heatmap_args()
    inspect_pangenome_heatmap_args
    return (inspect_pangenome_heatmap_args,)


@app.cell
def _(inspect_pangenome, inspect_pangenome_heatmap_args):
    inspect_pangenome.plot_heatmap(**inspect_pangenome_heatmap_args.value)
    return


@app.cell
def _(inspect_pangenome):
    inspect_pangenome_plot_scatter_args = inspect_pangenome.plot_scatter_args()
    inspect_pangenome_plot_scatter_args
    return (inspect_pangenome_plot_scatter_args,)


@app.cell
def _(inspect_pangenome, inspect_pangenome_plot_scatter_args):
    inspect_pangenome.plot_scatter(**inspect_pangenome_plot_scatter_args.value)
    return


@app.cell
def _(mo):
    mo.md(r"""## Compare Pangenomes""")
    return


@app.cell
def _(id_to_name, mo, name_to_id, pangenome_datasets, query_params):
    # Let the user select which pangenome datasets to compare
    compare_pangenomes_dataset_ui = mo.ui.multiselect(
        label="Select Pangenomes to Compare:",
        value=[
            id_to_name(pangenome_datasets, ds_id)
            for ds_id in query_params.get("compare_pangenomes_datasets", "").split(",")
            if len(ds_id) > 0
        ],
        options=name_to_id(pangenome_datasets),
        on_change=lambda i: query_params.set("compare_pangenomes_datasets", ",".join(i))
    )
    compare_pangenomes_dataset_ui
    return (compare_pangenomes_dataset_ui,)


@app.cell
def _(
    client,
    compare_pangenomes_dataset_ui,
    id_to_name,
    mo,
    pangenome_datasets,
    project_ui,
):
    # Stop if the user has not selected a dataset
    mo.stop(compare_pangenomes_dataset_ui.value is None)

    # Get the selected datasets
    compare_pangenomes_datasets = [
        (
            client
            .get_project_by_id(project_ui.value)
            .get_dataset_by_id(ds_id)
        )
        for ds_id in compare_pangenomes_dataset_ui.value
    ]

    mo.md("""
    Selected:

    {selected}
    """.format(
        selected="\n".join([
            "- " + id_to_name(pangenome_datasets, i)
            for i in compare_pangenomes_dataset_ui.value
        ])
    ))
    return (compare_pangenomes_datasets,)


@app.cell
def _(
    DataPortalDataset,
    List,
    Pangenome,
    make_pangenome,
    mo,
    pd,
    px,
    query_params,
):
    class ComparePangenomes:
        pg_list: List[Pangenome]

        def __init__(self, ds_list: List[DataPortalDataset]):
            self.pg_list = [
                make_pangenome(ds, float(query_params.get("min_prop", 0.5)))
                for ds in ds_list
            ]

            # Combine the genome annotations across all pangneomes
            self.genome_df = pd.concat([
                (
                    pg.adata
                    .obs
                    .reindex(columns=["n_genes", "monophyly"])
                    .assign(
                        pangenome=pg.ds.name
                    )
                )
                for pg in self.pg_list
            ])

        def args(self, title: str):
            return mo.md(
                "### Compare Pangenomes: " + title + """

     - Number of Bins: {nbins}
     - Figure Height: {height}

            """).batch(
                nbins=mo.ui.number(
                    start=2,
                    stop=100,
                    step=1,
                    value=30
                ),
                height=mo.ui.number(
                    start=100,
                    stop=10000,
                    step=10,
                    value=600
                )
            )

        def plot(self, nbins: int, height: int, y: str, ylabel: str, title: str):

            fig = px.histogram(
                self.genome_df,
                y=y,
                facet_col="pangenome",
                labels={y: ylabel},
                template="simple_white",
                nbins=nbins,
                height=height,
                title=title
            )
            fig.update_xaxes(matches=None)
            return fig
    return (ComparePangenomes,)


@app.cell
def _(ComparePangenomes, compare_pangenomes_datasets):
    compare_pangenomes = ComparePangenomes(compare_pangenomes_datasets)
    return (compare_pangenomes,)


@app.cell
def _(compare_pangenomes):
    compare_pangenomes_ngenes_args = compare_pangenomes.args("Genome Size")
    compare_pangenomes_ngenes_args
    return (compare_pangenomes_ngenes_args,)


@app.cell
def _(compare_pangenomes, compare_pangenomes_ngenes_args):
    compare_pangenomes.plot(
        y="n_genes",
        ylabel="# of Genes per Genome",
        title="Number of Genes per Genome",
        **compare_pangenomes_ngenes_args.value
    )
    return


@app.cell
def _(compare_pangenomes):
    compare_pangenomes_monophyly_args = compare_pangenomes.args("Monophyly")
    compare_pangenomes_monophyly_args
    return (compare_pangenomes_monophyly_args,)


@app.cell
def _(compare_pangenomes, compare_pangenomes_monophyly_args):
    compare_pangenomes.plot(
        y="monophyly",
        ylabel="Monophyly",
        title="Monophyly",
        **compare_pangenomes_monophyly_args.value
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Inspect Gene Bin""")
    return


@app.cell
def _(id_to_name, mo, name_to_id, pangenome_datasets, query_params):
    # Let the user select which pangenome dataset to get data from
    gene_bin_dataset_ui = mo.ui.dropdown(
        label="Select Pangenome:",
        value=id_to_name(pangenome_datasets, query_params.get("inspect_gene_bin_dataset")),
        options=name_to_id(pangenome_datasets),
        on_change=lambda i: query_params.set("inspect_gene_bin_dataset", i)
    )
    gene_bin_dataset_ui
    return (gene_bin_dataset_ui,)


@app.cell
def _(client, gene_bin_dataset_ui, mo, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(gene_bin_dataset_ui.value is None)

    # Get the selected dataset
    gene_bin_dataset = (
        client
        .get_project_by_id(project_ui.value)
        .get_dataset_by_id(gene_bin_dataset_ui.value)
    )
    return (gene_bin_dataset,)


@app.cell
def _(DataPortalDataset, Pangenome, make_pangenome, mo, query_params):
    class InspectGeneBin:
        pg: Pangenome

        def __init__(self, ds: DataPortalDataset):
            self.pg = make_pangenome(ds, float(query_params.get("min_prop", 0.5)))

        def display_bin_args(self):
            return mo.md(
                """
        {bin_id}
    
        {genome_columns}
                """
            ).batch(
                bin_id=mo.ui.dropdown(
                    label="Display:",
                    options=sorted(self.pg.adata.var_names, key=lambda bin_id: int(bin_id.split(" ")[-1])),
                    value="Bin 1"
                ),
                genome_columns=mo.ui.multiselect(
                    label="Show Columns:",
                    options=self.pg.adata.obs.columns.values,
                    value=self.pg.adata.obs.columns.values
                )
            )

        def display_bin(self, bin_id, genome_columns):
            # Show the list of genes in the bin
            gene_df = (
                self.pg.bin_contents[bin_id]
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
                self.pg.adata
                [
                    self.pg.adata
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
    return (InspectGeneBin,)


@app.cell
def _(InspectGeneBin, gene_bin_dataset):
    inspect_gene_bin = InspectGeneBin(gene_bin_dataset)
    return (inspect_gene_bin,)


@app.cell
def _(inspect_gene_bin):
    # Show the user information about a specific bin
    display_bin_args = inspect_gene_bin.display_bin_args()
    display_bin_args
    return (display_bin_args,)


@app.cell
def _():
    return


@app.cell
def _(display_bin_args, inspect_gene_bin):
    inspect_gene_bin.display_bin(**display_bin_args.value)
    return


@app.cell
def _(mo):
    mo.md(r"""## Compare Gene Bins""")
    return


@app.cell
def _(mo, pg):
    # Show the user overlap between bins
    bin_overlap_args = mo.md("""
    {bins}

    {height}

    """).batch(
        bins=mo.ui.multiselect(
            label="Bins:",
            options=sorted(pg.adata.var_names, key=lambda bin_id: int(bin_id.split(" ")[-1])),
            value=list(pg.adata.var.sort_values(by="n_genes", ascending=False).head(10).index.values)
        ),
        height=mo.ui.number(
            label="Figure Height:",
            start=100,
            value=800
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
                annot_df = annot_df.dropna(
                    axis=0,
                    how="all"
                )
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
def _(mo):
    mo.md(r"""## Settings""")
    return


@app.cell
def _(mo, query_params):
    mo.md("""
    ---

    The gig-map analysis will identify every individual gene which is encoded by each genome
    in the pan-genome collection.
    To determine whether a particular genome contains a particular gene bin (which generally consist
    of multiple genes), a minimum threshold is applied for the proportion of genes which must be detected
    for that bin to be counted as present.

    {min_prop}

    ---

    Genomes with highly fragmented assemblies may not contain a large number of completely-assembled
    genes from the pan-genome collection.
    To filter out these genomes, a lower bound is set on the number of genes that are contained in a
    genome.
    This threshold is set as a proportion relative to the median number of genes encoded by genomes
    in this dataset.

    {min_genes_rel_median}

    ---
    """).batch(
        min_prop=mo.ui.number(
            label="Minimum Proportion of Genes in Bin:",
            start=0.01,
            stop=1.0,
            value=float(query_params.get("min_prop", '0.5')),
            on_change=lambda val: query_params.set("min_prop", val)
        ),
        min_genes_rel_median=mo.ui.number(
            label="Minimum Number of Genes per Genome (Median Fraction):",
            start=0.01,
            stop=1.0,
            step=0.01,
            value=float(query_params.get("min_genes_rel_median", '0.5')),
            on_change=lambda val: query_params.set("min_genes_rel_median", val)
        )
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
