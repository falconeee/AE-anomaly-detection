import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
from scipy.stats import invgauss
from scipy.stats.distributions import chi2
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def select_training_period(df_dataset, timestamp):
    df_np = df_dataset.copy()
    time = df_np[timestamp].to_list()
    df_np = df_np.drop(timestamp, axis=1).copy()
    array_np = df_np.to_numpy()

    ##PCA
    scaler = StandardScaler()
    # Fit on training set only.
    array_np_std = scaler.fit_transform(array_np)
    cov = np.cov(array_np_std.T)
    u, s, vh = np.linalg.svd(cov)
    pca = PCA(0.95)
    pca.fit(array_np_std)
    pc = pca.transform(array_np_std)
    nc = pc.shape[1]
    s_diag = np.diag(s)
    s_pcs = s_diag[:nc, :nc]

    ##T2
    t2 = []
    for i in range(pc.shape[0]):
        termo1 = pc[i]
        termo2 = np.linalg.inv(s_pcs)
        termo3 = pc[i].T

        t2.append(termo1.dot(termo2).dot(termo3))
    M = pc.shape[1]
    N = pc.shape[0]
    F = scipy.stats.f.ppf(0.95, M, N - M)
    t2_lim = (M * (N - 1) / (N - M)) * F

    ##SPE
    spe = []
    for i in range(pc.shape[0]):
        rs = array_np_std[i].dot(u[:, nc - 1 :])
        termo1 = rs.T
        termo2 = rs
        spe.append(termo1.dot(termo2))
    teta1 = (s_diag[nc - 1 :]).sum()
    teta2 = (s_diag[nc - 1 :] ** 2).sum()
    teta3 = (s_diag[nc:-1, :] ** 3).sum()
    h0 = 1 - (2 * teta1 * teta3) / (3 * teta2**2)
    mu = 0.145462645553
    vals = invgauss.ppf([0, 0.999], mu)
    ca = invgauss.cdf(vals, mu)[1]
    spe_lim = teta1 * (
        (h0 * ca * np.sqrt(2 * teta2) / teta1)
        + 1
        + (teta2 * h0 * (h0 - 1)) / (teta1**2)
    ) ** (1 / h0)

    ##PHI
    phi = []
    for i in range(pc.shape[0]):
        phi.append((spe[i] / spe_lim) + (t2[i] / t2_lim))
    gphi = ((nc / t2_lim**2) + (teta2 / spe_lim**2)) / (
        (nc / t2_lim) + (teta1 / spe_lim)
    )
    hphi = ((nc / t2_lim) + (teta1 / spe_lim)) ** 2 / (
        (nc / t2_lim**2) + (teta2 / spe_lim**2)
    )
    chi2.ppf(0.975, df=2)
    phi_lim = gphi * chi2.ppf(0.99, hphi)
    df_t2 = pd.DataFrame(
        {
            "time": time,
            "t2": t2,
            "spe": spe,
            "phi": phi,
        }
    )

    df_t2["t2_lim"] = t2_lim
    df_t2["spe_lim"] = spe_lim
    df_t2["phi_lim"] = phi_lim

    df_t2["t2"] = df_t2["t2"].ewm(alpha=0.01).mean()
    df_t2["spe"] = df_t2["spe"].ewm(alpha=0.01).mean()
    df_t2["phi"] = df_t2["phi"].ewm(alpha=0.01).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t2["time"], y=df_t2["phi"], mode="lines"))
    fig.add_trace(go.Scatter(x=df_t2["time"], y=df_t2["phi_lim"], mode="lines"))

    return fig
    # Nome do arquivo Excel local
    file_name = "processos.xlsx"

    try:
        # Ler o arquivo Excel
        df_process = pd.read_excel(file_name)

        # Consultar o process_id correspondente
        process_id = df_process.loc[df_process['process_name'] == process_name, 'process_id'].values[0]
    
    except (FileNotFoundError, IndexError):
        # Se o arquivo não for encontrado ou o processo não estiver no arquivo
        process_id = None
    
    return process_id
