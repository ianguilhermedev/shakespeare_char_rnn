from pathlib import Path 
import urllib.request

def data_download():
    path = Path('../dataset/shakespeare/shakespeare.txt')

    #verifica se há algum arquivo nesse caminho definido
    if not path.is_file():
        path.parent.mkdir(parents=True, #cria pastas faltantes no caminho
                          exist_ok=True)
        
        url = "https://homl.info/shakespeare"
        urllib.request.urlretrieve(url, path)
    return path.read_text()
