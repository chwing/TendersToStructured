# Thin layer on the existing tender-pipeline image: adds what the benchmark needs
# (psutil for RAM, gliner for Staged 1). Code and documents are bind-mounted at run time.
FROM tender-pipeline:latest
RUN pip install --no-cache-dir psutil "gliner>=0.2.0"
