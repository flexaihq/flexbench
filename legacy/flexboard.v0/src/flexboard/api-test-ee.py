# Testing ElasticSearch
#
# Author(s): Grigori Fursin
# Contributor(s): Gauthier Wallet

import os
from fastapi import FastAPI
from elasticsearch import Elasticsearch

#########################################################
# Elastic Search configuration
ELASTIC_URL = 'https://flexboard-eck-elasticsearch-es-http.elastic-staging:9200'
#ELASTIC_URL = 'https://127.0.0.1:9200'

ELASTIC_API_KEY = os.environ['ELASTIC_API_KEY'] # provided via Kubernetes secrets

#########################################################
# Init
app = FastAPI()

#########################################################
# End-points
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/es")
async def test_es():
    try:
        es = Elasticsearch(
            ELASTIC_URL,
            verify_certs=False,  # Disables SSL certificate verification
            headers={"Authorization": f"ApiKey {ELASTIC_API_KEY}"},
#            basic_auth=("elastic", ""),
#            ssl_show_warn=False,   # Suppresses SSL warnings
#            client_cert="self-signed-es-client.crt",  # Add client certificate
#            client_key="self-signed-es-client.key"  # Add client private key
        )

        client_info = es.info()

        return {"message": str(client_info.body)}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
