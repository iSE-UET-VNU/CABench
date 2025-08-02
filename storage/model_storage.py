import json
import os
from typing import List, Dict, Union
from tqdm import tqdm
from utils.constants import (
    ROOT_PATH,
    EMBEDDING_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_API_TYPE,
)
from utils.common import singleton
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

@singleton
class ModelStorage(object):

    def __init__(self, cache=f"{ROOT_PATH}/cache/model_storage", reset=False):
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
            meta_data = self.load_meta_data()
            self.meta_storage = Chroma.from_documents(
                documents=meta_data,
                embedding=self.embedding,
                persist_directory=meta_cache,
            )

        model_cache = os.path.join(cache, "models")
        if os.path.exists(model_cache) and not reset:
            self.model_storage = Chroma(
                embedding_function=self.embedding,
                persist_directory=model_cache,
            )
        else:
            self.model_storage = Chroma(
                embedding_function=self.embedding, persist_directory=model_cache
            )
            self.load_model_information_batch(batch_size=50)

    def add_model(self, model_data: Dict) -> bool:
        """
        Thêm một model mới vào storage

        Args:
            model_data: Dict chứa thông tin model với format:
                {
                    "tag": "image-classification",
                    "id": "model-id",
                    "desc": "Model description",
                    "inference_type": "huggingface",
                    "meta": {} # optional
                }

        Returns:
            bool: True nếu thêm thành công, False nếu thất bại
        """
        try:
            # Format model description
            model_info, meta_data = self.format_model_desc(model_data)

            # Tạo document mới
            doc = Document(page_content=model_info, metadata=meta_data)

            # Thêm vào storage
            self.model_storage.add_documents([doc])

            # Lưu vào file JSONL
            self._append_model_to_file(model_data)

            return True
        except Exception as e:
            print(f"Lỗi khi thêm model: {e}")
            return False

    def add_metadata(self, metadata: Dict) -> bool:
        """
        Add new metadata to storage

        Args:
            metadata: Dict contains metadata with format:
                {
                    "id": "Node ID",
                    "type": "node-type",
                    "desc": "Description",
                    "input-type": ["text"],
                    "output-type": ["text"],
                    "inputs": [...], # optional
                    "outputs": [...] # optional
                }

        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Format metadata
            meta_str, meta_data = self.format_node_meta(metadata)

            # Create new document
            doc = Document(page_content=meta_str, metadata=meta_data)

            # Add to storage
            self.meta_storage.add_documents([doc])

            # Save to file JSON
            self._append_metadata_to_file(metadata)

            return True
        except Exception as e:
            print(f"Error while adding metadata: {e}")
            return False

    def add_models_batch(self, models: List[Dict]) -> List[bool]:
        """
        Thêm nhiều model cùng lúc

        Args:
            models: List các dict chứa thông tin model

        Returns:
            List[bool]: Kết quả thêm từng model
        """
        results = []
        for model in models:
            results.append(self.add_model(model))
        return results

    def add_metadata_batch(self, metadata_list: List[Dict]) -> List[bool]:
        """
        Thêm nhiều metadata cùng lúc

        Args:
            metadata_list: List các dict chứa thông tin metadata

        Returns:
            List[bool]: Kết quả thêm từng metadata
        """
        results = []
        for metadata in metadata_list:
            results.append(self.add_metadata(metadata))
        return results

    def _append_model_to_file(self, model_data: Dict):
        """Thêm model vào file JSONL"""
        file_path = f"{ROOT_PATH}/data/huggingface_models.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(model_data, ensure_ascii=False) + "\n")

    def _append_metadata_to_file(self, metadata: Dict):
        """Thêm metadata vào file JSON"""
        file_path = f"{ROOT_PATH}/data/meta.json"

        # Đọc file hiện tại
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Thêm metadata mới
        data["nodes"].append(metadata)

        # Ghi lại file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def remove_model(self, model_id: str) -> bool:
        """
        Xóa model theo ID

        Args:
            model_id: ID của model cần xóa

        Returns:
            bool: True nếu xóa thành công, False nếu thất bại
        """
        try:
            self._remove_model_from_file(model_id)
            return True
        except Exception as e:
            print(f"Lỗi khi xóa model: {e}")
            return False

    def remove_metadata(self, meta_id: str) -> bool:
        """
        Xóa metadata theo ID

        Args:
            meta_id: ID của metadata cần xóa

        Returns:
            bool: True nếu xóa thành công, False nếu thất bại
        """
        try:
            self._remove_metadata_from_file(meta_id)
            return True
        except Exception as e:
            print(f"Lỗi khi xóa metadata: {e}")
            return False

    def _remove_model_from_file(self, model_id: str):
        """Xóa model từ file JSONL"""
        file_path = f"{ROOT_PATH}/data/huggingface_models.jsonl"
        temp_file = f"{ROOT_PATH}/data/huggingface_models_temp.jsonl"

        with open(file_path, "r", encoding="utf-8") as f_in:
            with open(temp_file, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    model_data = json.loads(line.strip())
                    if model_data.get("id") != model_id:
                        f_out.write(line)

        os.replace(temp_file, file_path)

    def _remove_metadata_from_file(self, meta_id: str):
        """Xóa metadata từ file JSON"""
        file_path = f"{ROOT_PATH}/data/meta.json"

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Lọc bỏ metadata có ID trùng
        data["nodes"] = [node for node in data["nodes"] if node.get("id") != meta_id]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def update_model(self, model_id: str, new_data: Dict) -> bool:
        """
        Cập nhật thông tin model

        Args:
            model_id: ID của model cần cập nhật
            new_data: Dữ liệu mới

        Returns:
            bool: True nếu cập nhật thành công, False nếu thất bại
        """
        try:
            # Xóa model cũ
            self.remove_model(model_id)

            # Thêm model mới
            new_data["id"] = model_id
            return self.add_model(new_data)
        except Exception as e:
            print(f"Lỗi khi cập nhật model: {e}")
            return False

    def update_metadata(self, meta_id: str, new_data: Dict) -> bool:
        """
        Cập nhật thông tin metadata

        Args:
            meta_id: ID của metadata cần cập nhật
            new_data: Dữ liệu mới

        Returns:
            bool: True nếu cập nhật thành công, False nếu thất bại
        """
        try:
            # Xóa metadata cũ
            self.remove_metadata(meta_id)

            # Thêm metadata mới
            new_data["id"] = meta_id
            return self.add_metadata(new_data)
        except Exception as e:
            print(f"Lỗi khi cập nhật metadata: {e}")
            return False

    def get_model_by_id(self, model_id: str):
        """Tìm model theo ID"""
        return self.model_storage.get(where={"id": model_id})

    def search_model(self, model_desc: str, top_k: int = 20) -> List[Document]:
        docs = self.model_storage.similarity_search(model_desc, k=top_k)
        return docs

    def search_model_with_tag(
        self, model_desc: str, tag: Union[str, List[str]], top_k=20
    ) -> List[Document]:
        """Tìm model theo tag (có thể là string hoặc list of strings)"""
        docs = []
        if isinstance(tag, str):
            # Search for models where tags field contains the tag
            docs = self.model_storage.similarity_search(
                model_desc,
                filter={
                    "pipeline_tag": tag
                },  # Use pipeline_tag instead of tag for exact match
                k=top_k,
            )
        elif isinstance(tag, list):
            docs = self.model_storage.similarity_search(
                model_desc,
                filter={
                    "pipeline_tag": {"$in": tag}
                },  # Filter by multiple tags at once
                k=top_k,
            )

        return docs

    def hybrid_search_model_with_tag(
        self, 
        model_desc: str, 
        tag: Union[str, List[str]], 
        top_k: int = 20, 
        alpha: float = 0.7
    ) -> List[Document]:
        """
        Hybrid search kết hợp semantic search và keyword search với tag filtering
        
        Args:
            model_desc: Mô tả model cần tìm
            tag: Tag(s) để filter (string hoặc list of strings)
            top_k: Số lượng kết quả trả về
            alpha: Trọng số cho semantic search (0.0-1.0), (1-alpha) cho keyword search
            
        Returns:
            List[Document]: Danh sách documents đã được ranked và filtered theo tag
        """
        try:
            # 1. Semantic search với tag filter
            semantic_docs = []
            if isinstance(tag, str):
                semantic_docs = self.model_storage.similarity_search(
                    model_desc,
                    filter={"pipeline_tag": tag},
                    k=min(top_k * 2, 50)
                )
            elif isinstance(tag, list):
                semantic_docs = self.model_storage.similarity_search(
                    model_desc,
                    filter={"pipeline_tag": {"$in": tag}},
                    k=min(top_k * 2, 50)
                )
            
            # 2. Keyword search với tag filter
            keywords = self._extract_keywords(model_desc)
            keyword_docs = self._keyword_search_with_tag(keywords, tag, top_k * 2)
            
            # 3. Combine và rerank results
            combined_docs = self._combine_and_rerank(
                semantic_docs, keyword_docs, alpha, top_k
            )
            
            return combined_docs[:top_k]
            
        except Exception as e:
            print(f"Lỗi trong hybrid search with tag: {e}")
            # Fallback to normal search with tag
            return self.search_model_with_tag(model_desc, tag, top_k)
    
    def _keyword_search_with_tag(
        self, 
        keywords: List[str], 
        tag: Union[str, List[str]], 
        top_k: int
    ) -> List[Document]:
        """Tìm kiếm theo keywords với tag filter"""
        if not keywords:
            return []
            
        # Get all docs với tag filter
        if isinstance(tag, str):
            filtered_docs = self.model_storage.get(where={"pipeline_tag": tag})
        elif isinstance(tag, list):
            # For multiple tags, get docs for each tag
            all_filtered = []
            for single_tag in tag:
                tag_docs = self.model_storage.get(where={"pipeline_tag": single_tag})
                if tag_docs and 'documents' in tag_docs:
                    # Combine documents and metadatas
                    documents = tag_docs['documents']
                    metadatas = tag_docs.get('metadatas', [])
                    for i, doc in enumerate(documents):
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        all_filtered.append({
                            'document': doc,
                            'metadata': metadata
                        })
            
            # Convert back to expected format
            if all_filtered:
                filtered_docs = {
                    'documents': [item['document'] for item in all_filtered],
                    'metadatas': [item['metadata'] for item in all_filtered]
                }
            else:
                filtered_docs = {'documents': [], 'metadatas': []}
        else:
            return []
        
        if not filtered_docs or 'documents' not in filtered_docs:
            return []
            
        documents = filtered_docs['documents']
        metadatas = filtered_docs.get('metadatas', [])
        
        keyword_docs = []
        for i, doc_content in enumerate(documents):
            score = 0
            doc_lower = doc_content.lower()
            
            # Score dựa trên số lượng keywords match
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in doc_lower:
                    # Boost score cho exact match
                    score += 2 if f' {keyword_lower} ' in doc_lower else 1
                    
                # Check trong metadata nếu có
                if i < len(metadatas) and metadatas[i]:
                    metadata = metadatas[i]
                    for key, value in metadata.items():
                        if isinstance(value, str) and keyword_lower in value.lower():
                            score += 1.5
            
            if score > 0:
                doc = Document(
                    page_content=doc_content,
                    metadata=metadatas[i] if i < len(metadatas) else {}
                )
                doc.metadata['keyword_score'] = score
                keyword_docs.append(doc)
        
        # Sort by keyword score
        keyword_docs.sort(key=lambda x: x.metadata.get('keyword_score', 0), reverse=True)
        return keyword_docs[:top_k]

    def get_content_from_document(self, docs: Union[Document, List[Document]]) -> str:
        """Get content from search"""
        if isinstance(docs, Document):
            return docs.page_content
        content = []
        for doc in docs:
            content.append(doc.page_content)
        return "\n\n".join(content)

    def get_models_by_tag(self, tag: Union[str, List[str]]) -> List[Document]:
        """Tìm tất cả model có cùng tag (có thể là string hoặc list of strings)"""
        if isinstance(tag, str):
            # Search by pipeline_tag for exact match
            return self.model_storage.get(where={"pipeline_tag": tag})
        elif isinstance(tag, list):
            # For list of tags, collect all results
            all_docs = []
            for single_tag in tag:
                docs = self.model_storage.get(where={"pipeline_tag": single_tag})
                if docs and "documents" in docs:
                    all_docs.extend(docs["documents"])
            return all_docs
        else:
            raise ValueError("Tag must be string or list of strings")

    def get_meta_by_id(self, meta_id: str) -> List[Document]:
        """Tìm metadata theo ID"""
        return self.meta_storage.similarity_search(meta_id, filter={"id": meta_id})

    def get_meta_by_tag(self, tag: str, format: bool = True) -> List[Document]:
        """Tìm metadata theo tag"""
        docs = self.meta_storage.get(where={"tag": tag})
        if format:
            return "\n\n".join(docs["documents"])
        return docs

    def get_meta_by_input_type(self, input_type: str) -> List[Document]:
        """Tìm metadata theo input type"""
        return self.meta_storage.similarity_search(
            input_type, filter={"input_type": input_type}
        )

    def get_meta_by_output_type(self, output_type: str) -> List[Document]:
        """Tìm metadata theo output type"""
        return self.meta_storage.similarity_search(
            output_type, filter={"output_type": output_type}
        )

    def get_meta_by_input_output_type(
        self, input_type: str, output_type: str
    ) -> List[Document]:
        """Tìm metadata theo input và output type"""
        return self.meta_storage.similarity_search(
            f"{input_type}, {output_type}",
            filter={"$and": [{"input_type": input_type}, {"output_type": output_type}]},
        )

    def get_models_by_filter(self, filter: Dict) -> List[Document]:
        """Tìm model theo filter"""
        return self.model_storage.get(**filter)

    def get_all_models(self) -> List[str]:
        """Lấy tất cả model information"""
        return self.model_storage.get()

    def get_all_meta_data(self) -> List[str]:
        """Lấy tất cả metadata"""
        return self.meta_storage.get()

    def demo_hybrid_search(self, query: str, tag: str = "image-classification"):
        """
        Demo để so sánh kết quả giữa semantic search và hybrid search
        
        Args:
            query: Câu query để test
            tag: Tag để filter (mặc định là image-classification)
        """
        print(f"\n🔍 Demo Hybrid Search vs Semantic Search")
        print(f"Query: '{query}'")
        print(f"Tag filter: '{tag}'")
        print("="*60)
        
        # 1. Semantic search only
        print("\n📊 SEMANTIC SEARCH ONLY:")
        semantic_results = self.search_model_with_tag(query, tag, top_k=5)
        for i, doc in enumerate(semantic_results[:3], 1):
            model_id = doc.metadata.get('id', 'Unknown')
            pipeline_tag = doc.metadata.get('pipeline_tag', 'Unknown')
            print(f"{i}. {model_id} ({pipeline_tag})")
            
        # 2. Hybrid search
        print(f"\n🔥 HYBRID SEARCH (alpha=0.7):")
        hybrid_results = self.hybrid_search_model_with_tag(query, tag, top_k=5, alpha=0.7)
        for i, doc in enumerate(hybrid_results[:3], 1):
            model_id = doc.metadata.get('id', 'Unknown')
            pipeline_tag = doc.metadata.get('pipeline_tag', 'Unknown')
            semantic_score = doc.metadata.get('semantic_score', 0)
            keyword_score = doc.metadata.get('keyword_score', 0)
            combined_score = doc.metadata.get('combined_score', 0)
            print(f"{i}. {model_id} ({pipeline_tag})")
            print(f"   📈 Scores: semantic={semantic_score:.3f}, keyword={keyword_score:.3f}, combined={combined_score:.3f}")
            
        # 3. Keywords extracted
        keywords = self._extract_keywords(query)
        print(f"\n🔑 Keywords extracted: {keywords}")
        
        return {
            'semantic_results': semantic_results,
            'hybrid_results': hybrid_results,
            'keywords': keywords
        }

    def format_model_desc(self, model: Dict) -> str:
        desc_str = f'Model "{model["id"]}":\n'
        desc_str += f'- Model type: {model["pipeline_tag"]}\n'
        desc_str += f'- Inference type: {model["inference_type"]}\n'

        # Convert lists to strings for Chroma metadata compatibility
        tags_str = (
            ",".join(model["tags"])
            if isinstance(model["tags"], list)
            else str(model["tags"])
        )
        language_str = None
        datasets_str = None

        if "language" in model["meta"] and model["meta"]["language"]:
            if isinstance(model["meta"]["language"], list):
                language_str = ",".join(model["meta"]["language"])
            else:
                language_str = str(model["meta"]["language"])

        if "datasets" in model["meta"] and model["meta"]["datasets"]:
            if isinstance(model["meta"]["datasets"], list):
                datasets_str = ",".join(model["meta"]["datasets"])
            else:
                datasets_str = str(model["meta"]["datasets"])

        meta_data = {
            "id": model["id"],
            "pipeline_tag": model["pipeline_tag"],
            "tags": tags_str,  # Convert list to comma-separated string
            "downloads": model["downloads"],
            "likes": model["likes"],
            "inference_type": model["inference_type"],
            "meta": json.dumps(model["meta"], ensure_ascii=False),
            "language": language_str,  # Convert list to comma-separated string
            "datasets": datasets_str,  # Convert list to comma-separated string
        }

        if "language" in model["meta"]:
            desc_str += f"- Supporting Language: {model['meta']['language']}\n"

        if "datasets" in model["meta"]:
            desc_str += f"- Trained on Datasets: {model['meta']['datasets']}\n"
        desc_str += f"- Tags :  {tags_str}\n"
        desc_str += f"- Description: {model['description']}\n"
        return desc_str, meta_data

    def load_model_information_batch(self, batch_size: int = 50):
        """Load and add model information in batches to avoid token limit"""
        print(f"Loading model information in batches of {batch_size}...")

        batch = []
        batch_count = 0

        with open(
            f"{ROOT_PATH}/data/huggingface_models.jsonl", "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        for i, line in enumerate(tqdm(lines, desc="Processing models", ncols=80)):
            try:
                model_info, meta_data = self.format_model_desc(json.loads(line))
                batch.append(Document(page_content=model_info, metadata=meta_data))

                # When batch is full or it's the last item, add to storage
                if len(batch) >= batch_size or i == len(lines) - 1:
                    batch_count += 1
                    print(f"Adding batch {batch_count} with {len(batch)} models...")

                    try:
                        self.model_storage.add_documents(batch)
                        print(f"✅ Batch {batch_count} added successfully")
                    except Exception as e:
                        print(f"❌ Error adding batch {batch_count}: {e}")
                        # If token limit error, try progressive splitting
                        if "max_tokens_per_request" in str(e):
                            print(
                                f"🔄 Token limit exceeded, splitting batch of {len(batch)} documents..."
                            )
                            self._add_batch_with_progressive_split(batch, batch_count)
                        else:
                            raise e

                    batch = []  # Reset batch

            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue

    def _add_batch_with_progressive_split(
        self, batch: List[Document], batch_number: int
    ):
        """Progressive splitting when hitting token limits"""
        batch_size = len(batch)

        # Try different split factors: 2, 3, 4, 5, 10
        split_factors = [2, 3, 4, 5, 10]

        for factor in split_factors:
            chunk_size = max(1, batch_size // factor)
            print(
                f"🔄 Trying {factor}-way split (chunks of ~{chunk_size} documents)..."
            )

            # Split batch into chunks
            success_count = 0
            all_chunks_success = True

            for i in range(0, batch_size, chunk_size):
                chunk = batch[i : i + chunk_size]
                try:
                    self.model_storage.add_documents(chunk)
                    success_count += len(chunk)
                    print(
                        f"✅ Added chunk {i//chunk_size + 1}/{(batch_size + chunk_size - 1)//chunk_size} ({len(chunk)} docs)"
                    )
                except Exception as chunk_e:
                    if "max_tokens_per_request" in str(chunk_e):
                        print(f"⚠️  Chunk still too large, will try smaller split...")
                        all_chunks_success = False
                        break
                    else:
                        print(f"❌ Chunk error (non-token): {chunk_e}")
                        all_chunks_success = False
                        break

            if all_chunks_success:
                # All chunks processed successfully
                print(
                    f"✅ Batch {batch_number} completed with {factor}-way split ({success_count}/{batch_size} documents)"
                )
                return

        # If all split attempts fail, try adding one by one
        print("🔄 All batch splits failed, trying individual documents...")
        success_count = 0
        for i, doc in enumerate(batch):
            try:
                self.model_storage.add_documents([doc])
                success_count += 1
                if (i + 1) % 5 == 0:  # Progress update every 5 docs
                    print(f"📄 Added {i + 1}/{len(batch)} individual documents...")
            except Exception as single_e:
                print(f"❌ Failed to add individual document {i}: {single_e}")

        print(
            f"✅ Batch {batch_number} completed individually ({success_count}/{len(batch)} documents)"
        )

    def load_model_information(self):
        models = []
        with open(
            f"{ROOT_PATH}/data/huggingface_models.jsonl", "r", encoding="utf-8"
        ) as f:
            for line in tqdm(f.readlines(), desc="Loading model information", ncols=80):
                model_info, meta_data = self.format_model_desc(json.loads(line))
                models.append(Document(page_content=model_info, metadata=meta_data))
        return models

    def format_node_meta(self, node: Dict) -> str:
        """Format a single node metadata into string format"""
        meta_str = f'Model "{node["id"]}":\n'
        meta_str += f'Description: {node["desc"]}\n'

        # Input types
        meta_str += f'Input types: {", ".join(node["input-type"])}\n'

        # Output types
        meta_str += f'Output types: {", ".join(node["output-type"])}\n'

        meta_data = {
            "id": node["id"],
            "tag": node["type"],
            "input_type": (
                ",".join(node["input-type"])
                if isinstance(node["input-type"], list)
                else str(node["input-type"])
            ),
            "output_type": (
                ",".join(node["output-type"])
                if isinstance(node["output-type"], list)
                else str(node["output-type"])
            ),
        }

        # Input format
        if "inputs" in node and node["inputs"]:
            meta_str += "Inputs format:\n"
            for input_field in node["inputs"]:
                meta_str += f"  - {input_field['name']} ({input_field['type']}): {input_field['description']}\n"

        # Output format
        if "outputs" in node and node["outputs"]:
            meta_str += "Outputs format:\n"
            for output_field in node["outputs"]:
                meta_str += f"  - {output_field['name']} ({output_field['type']})"
                if "items" in output_field:
                    meta_str += f": {json.dumps(output_field['items'], indent=2)}"
                if "description" in output_field:
                    meta_str += f": {output_field['description']}"
                meta_str += "\n"

        return meta_str, meta_data

    def load_meta_data(self):
        with open(f"{ROOT_PATH}/data/meta.json", "r") as f:
            data = json.load(f)
        meta_datas = []
        for node in tqdm(data["nodes"], desc="Loading metadata", ncols=80):
            meta_data_str, meta_data = self.format_node_meta(node)
            meta_datas.append(Document(page_content=meta_data_str, metadata=meta_data))
        return meta_datas


model_storage = ModelStorage()

if __name__ == "__main__":
    model_storage.add_model(
        {
            "tag": "image-classification",
            "id": "prithivMLmods/Mnist-Digits-SigLIP2",
            "desc": "MNIST digit classification model based on SigLIP2 architecture, designed for RGB image input. This model can handle color images and provides robust digit recognition capabilities with enhanced visual understanding through the SigLIP2 framework.",
            "inference_type": "local",
        }
    )
