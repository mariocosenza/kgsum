FROM continuumio/miniconda3:24.11.1-0

WORKDIR /home/docker_conda_template

# Copy the boot script from docker/ml
COPY docker/ml/boot.sh ./

# Copy project files from the context root
COPY src ./src
COPY frontend ./frontend
COPY data/trained ./data/trained
COPY environment.yml .
COPY app.py .
COPY config.py .

RUN chmod +x boot.sh

RUN conda env create --verbose -f environment.yml

# Install spaCy models after creating the environment
RUN /opt/conda/envs/myenv/bin/python -m spacy download de_dep_news_trf \
 && /opt/conda/envs/myenv/bin/python -m spacy download en_core_web_trf \
 && /opt/conda/envs/myenv/bin/python -m spacy download es_dep_news_trf \
 && /opt/conda/envs/myenv/bin/python -m spacy download it_core_news_lg \
 && /opt/conda/envs/myenv/bin/python -m spacy download ja_core_news_trf \
 && /opt/conda/envs/myenv/bin/python -m spacy download nl_core_news_lg \
 && /opt/conda/envs/myenv/bin/python -m spacy download pt_core_news_lg \
 && /opt/conda/envs/myenv/bin/python -m spacy download ru_core_news_lg \
 && /opt/conda/envs/myenv/bin/python -m spacy download fr_dep_news_trf \
 && /opt/conda/envs/myenv/bin/python -m spacy download xx_sent_ud_sm \
 && /opt/conda/envs/myenv/bin/python -m spacy download zh_core_web_trf

ENV PATH="/opt/conda/envs/myenv/bin:$PATH"

EXPOSE 5000

ENTRYPOINT ["./boot.sh"]