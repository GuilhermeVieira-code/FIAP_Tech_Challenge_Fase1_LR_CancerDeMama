# Imagem base Python 3.9
FROM python:3.9-slim

# Informações do projeto
LABEL maintainer="Tech Challenge - Fase 1"
LABEL description="Sistema de Suporte ao Diagnóstico de Câncer de Mama"

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências primeiro (para aproveitar cache do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos os arquivos do projeto
COPY . .

# Expor porta do Jupyter Notebook
EXPOSE 8888

# Configurar Jupyter para aceitar conexões externas
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
