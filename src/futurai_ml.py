import numpy as np
import pandas as pd
from scipy.stats import f, norm
from scipy.stats.distributions import chi2
import math
from typing import Dict
import json


class FuturaiML:
    """
    Classe para geração do modelo futurai
    """

    def __init__(self, nc: int = 0, gain: int = 1):
        self.gain = gain
        self.nc = nc

    def fit(self, x_train: pd.DataFrame):
        """Função para treinamento do modelo
        :param - data - dataframe com a dataset de treinamento
        :return none
        """

        # Faz o scalling da base
        mean_train = x_train.mean()
        std_train = x_train.std()
        
        df = (x_train - mean_train) / std_train
        self.media = mean_train
        self.std = std_train

        # Aplicação do PCA para redução da quantidade de features do dataset
        linhas_colunas = df.shape

        # Matriz covariança dos dados
        df_array = np.array(df.T)
        matrix_cov = np.cov(df_array)

        # SVD para decomposiação da matriz covariança
        coeff, s, _ = np.linalg.svd(matrix_cov)
        coeff = pd.DataFrame(coeff)

        # Metodo VRE - Calculo das componentes principais
        if self.nc == 0:

            eps_pca = np.eye(linhas_colunas[1])
            vre = []

            for j in range(linhas_colunas[1]):

                # Calculo da C1
                residual = coeff.iloc[:, j : linhas_colunas[1]]
                val3_c1 = residual.dot(residual.T)

                val_ui = []
                for i in range(linhas_colunas[1]):
                    eps_aux = eps_pca[:, i]
                    eps_til = val3_c1.dot(eps_aux.T)
                    aux = (eps_til.T.dot(matrix_cov).dot(eps_til)) / (
                        eps_til.T.dot(eps_til) ** 2
                    )

                    val_ui.append(aux)

                vre.append(sum(val_ui) / (eps_aux.T.dot(matrix_cov).dot(eps_aux)))

            self.nc = vre.index(min(vre))

        # Calculo do PCA
        componentes_principais = coeff.iloc[:, 0 : self.nc]
        residual = coeff.iloc[:, self.nc : linhas_colunas[1]]
        aux_s = np.diag(s)
        df_s = pd.DataFrame(aux_s)
        val1 = df_s.iloc[: self.nc, : self.nc]
        val2_d = componentes_principais.dot(np.linalg.inv(val1)).dot(
            componentes_principais.T
        )
        val3_c1 = residual.dot(residual.T)
        self.c2 = componentes_principais.dot(componentes_principais.T)

        # Componentes principais
        coeff = coeff.iloc[:, 0 : self.nc]
        coeff = np.array(coeff)
        df = np.array(df)
        principal_components = df.dot(coeff)
        principal_components = pd.DataFrame(principal_components)

        self.val_d = val2_d
        self.val_c1 = val3_c1

        # Gera calculo dos limiares
        base_dados = principal_components
        a = self.nc
        ds = s

        alfa = 0.99
        n = base_dados.shape
        n = n[0]

        # Limiar da t2
        t2_lim = (a * (n - 1) * (n + 1) / (n * (n - a))) * f.ppf(alfa, a, n - a)

        # Limiar da Q
        teta1 = sum(ds[a:])
        teta2 = sum(ds[a:] ** 2)
        teta3 = sum(ds[a:] ** 3)

        h0 = 1 - (2 * teta1 * teta3) / (3 * (teta2 ** 2))
        ca = norm.ppf(alfa, 0, 1)
        q_lim = teta1 * (
            (h0 * ca * (math.sqrt(2 * teta2)) / teta1)
            + 1
            + (teta2 * h0 * (h0 - 1)) / (teta1 ** 2)
        ) ** (1 / h0)

        # Limiar phi
        gphi = ((a / t2_lim ** 2) + (teta2 / q_lim ** 2)) / (
            (a / t2_lim) + (teta1 / q_lim)
        )
        hphi = ((a / t2_lim) + (teta1 / q_lim)) ** 2 / (
            (a / t2_lim ** 2) + (teta2 / q_lim ** 2)
        )

        phi_lim = gphi * chi2.ppf(alfa, hphi)

        self.t2_lim = t2_lim
        self.q_lim = q_lim
        self.phi_lim = phi_lim * self.gain

    def predict(self, x_test, eixo_x, points=2) -> Dict:
        """Realiza a predição da base de dados
        :param - x_test: Base de dados para Predição
        :return - phi_matrix: uma matriz"""

        base = (x_test - self.media) / self.std

        abase = np.array(base)
        aval_d = self.val_d
        aval_c1 = self.val_c1
        t2_lim = self.t2_lim
        q_lim = self.q_lim

        # Estatistica Phi
        phi = []
        phi_matrix = (aval_d / t2_lim) + (aval_c1 / q_lim)

        for i in range(len(base)):
            phi.append(float((abase[i, :].T).dot(phi_matrix).dot(abase[i, :])))

        # Filtro
        dataset = pd.DataFrame([list(eixo_x), phi], index=["TIMESTAMP", "PHI"]).T
        dataset["TIMESTAMP"] = pd.to_datetime(
            dataset["TIMESTAMP"], format="%Y/%m/%d %H:%M:%S"
        )
        df_aux = dataset.copy()

        df_aux["status"] = 1
        ############################   Subida  ########################################
        ## Essa parte do código serve para pegar picos onde o motor volta a "funcionar"
        ## por menos de uma hora, ou seja, ele estava desligado, deu um pique de menos
        ## de uma hora e voltou a ficar desligado
        ###############################################################################
        data_aux = df_aux["TIMESTAMP"].min()  # Primeira data do dataframe
        while True:

            # Pega a data da primeira amostra com o valor abaixo do limite
            df_amostra = df_aux[
                (df_aux["PHI"] > self.phi_lim) & (df_aux["TIMESTAMP"] >= data_aux)
            ]

            if not df_amostra.empty:
                data_min = df_amostra["TIMESTAMP"].min()
            else:
                break

            # Pega a primeira data da amostra acima do valor limite depois da amostra acima
            df_amostra = df_aux[
                (df_aux["PHI"] <= self.phi_lim) & (df_aux["TIMESTAMP"] > data_min)
            ]

            if not df_amostra.empty:
                data_aux = df_amostra["TIMESTAMP"].min()

                mask = (df_aux["TIMESTAMP"] >= data_min) & (
                    df_aux["TIMESTAMP"] < data_aux
                )

                df_amostra = df_aux.loc[mask]

                if len(df_amostra) <= points:
                    df_aux["status"].loc[mask] = 0

            else:
                data_aux = df_aux["TIMESTAMP"].max()

                mask = (df_aux["TIMESTAMP"] >= data_min) & (
                    df_aux["TIMESTAMP"] <= data_aux
                )

                df_amostra = df_aux.loc[mask]

                if len(df_amostra) <= points:
                    df_aux["status"].loc[mask] = 0

                break

        dataset.drop(df_aux[df_aux["status"] == 0].index, inplace=True)

        phi = list(dataset["PHI"])
        eixo_x = list(dataset["TIMESTAMP"].dt.strftime("%Y-%m-%d %X"))

        predicao = {"matrix": phi_matrix, "phi": phi, "timestamp": eixo_x}

        return predicao

    def contribuition(self, df, phi, df_sistema, eixo_x, eixo_x_proj=None):
        """Função para gerar o grafico os de Contribuição e
        também lista das varaiveis que mais influênciaram"""

        try:
            df = (df - self.media) / self.std
            linhas_colunas = df.shape
            M = linhas_colunas[1]

            df_np = df.values
            if isinstance(phi, pd.DataFrame):
                phi_np = phi.values
            else:
                phi_np = phi

            # Otimização: Vetorização completa da geração da matriz de contribuição e RCI
            # Substitui o loop for e a inversão de matrizes custosas
            df_phi = df_np @ phi_np
            diag_phi = np.diag(phi_np)

            # Reconstrução das variáveis em falta (n_fast_list)
            n_fast_list_np = df_np - (df_phi / diag_phi)
            n_fast_list = pd.DataFrame(n_fast_list_np, columns=df.columns)

            # RCI contém o score de importância de cada variável (calculado de forma vetorizada)
            rci = np.sum(df_phi**2, axis=0) / diag_phi

            # Component RCI (circi) - Correção de bug do código original onde utilizava 
            # o escalar eps da última iteração para todas as colunas
            circi_np = (df_phi**2) / diag_phi
            circi = pd.DataFrame(circi_np, columns=df.columns)

            df_sistema["score"] = 0

            # Monta um dataframe de forma decrescente das variáveis conforme seu score
            df_rci = pd.DataFrame({"score": rci, "variavel": df.columns})
            df_rci = df_rci.sort_values(by="score", ascending=False)

            # Otimização: Mapeamento de scores direto no df_sistema ao invés de usar iterrows e loc
            score_dict = dict(zip(df_rci["variavel"], df_rci["score"]))
            df_sistema["score"] = df_sistema["VARIAVEL"].map(score_dict).fillna(0)

            df_score_dec = df_sistema.sort_values(by="score", ascending=False)

            # Recria a phi tirando um a um as varaiveis que mais influenciam até o phi ficar abaixo do limiar
            val_contr = []
            qtd_aux = 1

            for i, row in df_rci.iterrows():

                val_contr.append(i)

                eps = np.eye(M)[:, val_contr]
                phi_eps = eps.T @ phi_np @ eps
                inv_phi_eps = np.linalg.inv(phi_eps)

                # Definindo Trci - matriz diagonal com elementos um para as variáveis defeituosas
                I_trci = np.eye(M)
                for x_idx in val_contr:
                    I_trci[x_idx, x_idx] = 0

                # Reconstrução das variáveis em falta
                phi_sub = phi_np[val_contr, :]
                phi_sub_clean = phi_sub @ I_trci
                
                n_fast_np = - inv_phi_eps @ phi_sub_clean @ df_np.T
                n_fast_np = n_fast_np.T

                # Otimização: Cálculo vetorizado de phiast para todas as linhas (substitui loop de N iterações)
                df_clean = df_np @ I_trci
                part1 = np.sum((df_clean @ phi_np) * df_clean, axis=1)
                part2 = np.sum((n_fast_np @ phi_eps) * n_fast_np, axis=1)
                phiast = part1 - part2

                if np.max(phiast) < self.phi_lim and qtd_aux >= 3:
                    break

                # Quantidade de varaiveis que mais influenciaram
                qtd_aux = qtd_aux + 1

            # Separa as varaiveis que mais influenciam das restantes
            df_score_prin = df_score_dec.iloc[0:qtd_aux].copy()
            df_score_prin.reset_index(inplace=True, drop=True)

            ##### Gera dataframe com a projeção das variáveis PRINCIPAIS #####
            df_projection_prin_vars = pd.DataFrame(n_fast_np)
            df_projection_prin_vars.columns = list(df_score_prin["VARIAVEL"])
            df_projection_prin_vars = (df_projection_prin_vars * self.std) + self.media
            df_projection_prin_vars = df_projection_prin_vars[list(df_score_prin["VARIAVEL"])]
            
            # Resample Dataframe
            if eixo_x_proj is not None:
                df_projection_prin_vars['timestamp'] = list(eixo_x_proj)
                df_projection_prin_vars = df_projection_prin_vars.set_index('timestamp')
                df_projection_prin_vars = df_projection_prin_vars.resample('1T').asfreq()
                df_projection_prin_vars.reset_index(inplace=True)
                if 'timestamp' in df_projection_prin_vars.columns:
                    df_projection_prin_vars.drop('timestamp', inplace=True, axis=1)

            ##### Gera dataframe com a projeção de todas as VARIAVEIS #####
            df_projection_full = n_fast_list.copy()
            df_projection_full = (df_projection_full * self.std) + self.media
            # Resample Dataframe
            if eixo_x_proj is not None:
                df_projection_full['timestamp'] = list(eixo_x_proj)
                df_projection_full = df_projection_full.set_index('timestamp')
                df_projection_full = df_projection_full.resample('1T').asfreq()
                df_projection_full.reset_index(inplace=True)
                if 'timestamp' in df_projection_full.columns:
                    df_projection_full.drop('timestamp', inplace=True, axis=1)

            df_score_res = df_score_dec.iloc[qtd_aux:].copy()

            ######## Geração do grafico hierarquico conforme local ########
            soma_prin = df_score_prin["score"].sum()
            df_score_prin["%"] = df_score_prin.score.apply(
                lambda x: round((x / soma_prin * 100), 5) if soma_prin > 0 else 0
            )

            soma_dec = df_score_dec["score"].sum()
            df_score_dec["%"] = df_score_dec.score.apply(
                lambda x: round((x / soma_dec * 100), 5) if soma_dec > 0 else 0
            )

            # Geração do grafico de contribuição - As variaiveis que menos influenciaram são zeradas para não poluir o grafico
            df_contribuicao = circi.copy()
            for x in df_score_res["VARIAVEL"]:
                df_contribuicao.loc[:, x] = 0

            ######## Geração do grafico de contribuição ########
            df_contribuicao = df_contribuicao.join(
                pd.Series(list(eixo_x)).rename("timestamp"), how="right"
            )

            df_contribuicao_json = df_contribuicao.to_json(orient="columns")
            df_score_prin_json = df_score_prin.to_json(orient="columns")
            df_score_dec_json = df_score_dec.to_json(orient="columns")

            return (
                json.loads(df_score_prin_json),
                json.loads(df_contribuicao_json),
                json.loads(df_score_dec_json),
                df_projection_prin_vars
            )

        except ValueError as err:
            print(err)
            raise err