from . import tratar_texto, tratar_label
import pandas as pd


def filter_data(data_path, norel_path):
    # Nome do arquivo csv a ler
    nomeArquivoDadosBrutos = data_path
    # Carrega os dados na variavel 'data' utilizando o Pandas
    data = pd.read_csv(nomeArquivoDadosBrutos,
                       encoding="utf-8", low_memory=False)
    del nomeArquivoDadosBrutos
    # Trata o nome das colunas para trabalhar melhor com os dados
    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
    data.columns = [tratar_texto.removerCaracteresEspeciais(
        c)for c in data.columns]
    data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
    # Excluindo empenhos diferentes aglomerados na classe 92
    exercicio_anterior = data['natureza_despesa_cod'].str.contains(
        ".\..\...\.92\...", regex=True, na=False)
    index = exercicio_anterior.where(exercicio_anterior == True).dropna().index
    data.drop(index, inplace=True)
    data.reset_index(drop=True, inplace=True)
    # Deletando empenhos sem relevancia devido ao saldo zerado
    index = data["valor_saldo_do_empenho"].where(
        data["valor_saldo_do_empenho"] == 0).dropna().index
    data.drop(index, inplace=True)
    data.reset_index(drop=True, inplace=True)
    # data = data[:1000]  # limitando os dados para fazer testes
    # Deleta colunas que atraves de analise foram identificadas como nao uteis
    data = data.drop(['empenho_sequencial_empenho.1', 'classificacao_orcamentaria_descricao',
                      'natureza_despesa_nome', 'valor_estorno_anulacao_empenho',
                      'valor_anulacao_cancelamento_empenho', 'fonte_recurso_cod',
                      'elemento_despesa', 'grupo_despesa', 'empenho_sequencial_empenho'], axis='columns')
    # Funcao que gera o rotulo e retorna as linhas com as naturezas de despesa que so aparecem em 1 empenho
    label, linhas_label_unica = tratar_label.tratarLabel(data)
    label = pd.DataFrame(label)
    # Excluindo as naturezas de despesas que so tem 1 empenho
    data = data.drop(linhas_label_unica)
    data.reset_index(drop=True, inplace=True)
    del linhas_label_unica
    # Excluindo empenhos irrelevantes devido nao estarem mais em vigencia
    sem_relevancia = pd.read_excel(norel_path)
    sem_relevancia = sem_relevancia['Nat. Despesa']
    sem_relevancia = pd.DataFrame(sem_relevancia)

    excluir = []

    for i in range(len(sem_relevancia['Nat. Despesa'])):
        excluir.append(label.where(
            label['natureza_despesa_cod'] == sem_relevancia['Nat. Despesa'].iloc[i]).dropna().index)

    excluir = [item for sublist in excluir for item in sublist]
    # Excluindo as naturezas que nao estao mais vigentes
    label.drop(excluir, inplace=True)
    label.reset_index(drop=True, inplace=True)
    data.drop(excluir, inplace=True)
    data.reset_index(drop=True, inplace=True)
    del excluir, sem_relevancia

    return data
