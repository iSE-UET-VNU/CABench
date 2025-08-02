import os
import json
from typing import Dict
from utils.constants import ROOT_PATH
from utils.constants import (
    ROOT_PATH, EMBEDDING_BASE_URL, 
    EMBEDDING_API_KEY, EMBEDDING_MODEL_NAME
)
from utils.common import singleton
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

@singleton
class ResourceStorage(object):
    
    def __init__(self, cache = f"{ROOT_PATH}/cache/resource_storage", reset = False):
        self.embedding = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY,
        )
        
        if reset:
            import shutil
            if os.path.exists(cache):
                shutil.rmtree(cache)
        
        os.makedirs(cache, exist_ok=True)
        
        meta_cache = os.path.join(cache, "meta")
        if os.path.exists(meta_cache) and not reset:
            self.meta_storage = Chroma(
                embedding_function=self.embedding,
                persist_directory=meta_cache,
            )
        else:
            resource_desc = self.load_resource_desc()
            self.meta_storage = Chroma.from_documents(
                documents=resource_desc,
                embedding=self.embedding,
                persist_directory=meta_cache
            )
            
    def format_resource_desc(self, resource_desc: Dict) -> str:
        """Format resource description into string format"""
        meta_str = f'Resource "{resource_desc["name"]}":\n'
        meta_str += f'Description: {resource_desc["desc"]}\n'
        return meta_str
        
    def load_resource_desc(self):
        with open(f"{ROOT_PATH}/data/resource_desc.json", "r") as f:
            data = json.load(f)
        return data

resource_storage = ResourceStorage()
