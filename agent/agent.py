import os
import json
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import torch

class DocumentStore:
    """Handles document ingestion and storage."""
    
    def __init__(self, docs_dir: str = 'docs'):
        self.docs_dir = docs_dir
        self.documents = []
        self.doc_metadata = []
        
    def load_documents(self) -> List[Dict]:
        """Load all markdown documents from the docs directory."""
        
        doc_files = [
            'pricing.md',
            'refund_policy.md', 
            'product_specs_widget.md',
            'onboarding_guide.md',
            'troubleshooting.md',
            'release_notes.md'
        ]
        
        for filename in doc_files:
            filepath = os.path.join(self.docs_dir, filename)
            
            # Handle both actual files and simulated content
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # Use the content from the provided documents
                content = self._get_simulated_content(filename)
            
            # Split into chunks (simple paragraph-based splitting)
            chunks = self._split_into_chunks(content)
            
            for chunk in chunks:
                if chunk.strip():  # Skip empty chunks
                    self.documents.append(chunk)
                    self.doc_metadata.append({
                        'source': filename,
                        'content': chunk
                    })
        
        print(f"📚 Loaded {len(self.documents)} text chunks from {len(doc_files)} documents")
        return self.doc_metadata
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into manageable chunks."""
        
        # Split by double newlines (paragraphs) first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_simulated_content(self, filename: str) -> str:
        """Return simulated content based on the provided documents."""
        
        content_map = {
            'pricing.md': """# Pricing & Discounts
- Widget A: MSRP ₹6,999; Widget B: MSRP ₹9,499.
- Gizmo Lite: MSRP ₹12,999; Gizmo Pro: MSRP ₹19,999.
- Services have tiered discounts:
  - Service Basic: up to 10% discount for annual billing.
  - Service Plus: up to 15% discount for annual billing.
Bulk discounts apply for orders ≥ 20 units (contact sales).""",
            
            'refund_policy.md': """# Refund & Returns
- Return window is 30 days from delivery.
- Refunds are issued to the original payment method.
- Service subscriptions are refundable within 14 days of purchase if under 2 hours of usage.
- WhatsApp purchases are eligible for the same policy as Web.""",
            
            'product_specs_widget.md': """# Widget Specifications
- Widget A: battery life 12h, weight 210g, supports Bluetooth 5.2.
- Widget B: battery life 20h, weight 240g, supports Bluetooth 5.3, IP67.""",
            
            'onboarding_guide.md': """# Onboarding Guide
1. Create an account.
2. Verify email and phone.
3. Install the mobile or desktop client.
4. Connect the WhatsApp channel if needed.""",
            
            'troubleshooting.md': """# Troubleshooting
- If login fails, reset the password and check 2FA.
- For install errors, clear cache and re-run with admin rights.
- For WhatsApp connection issues, ensure your phone has network connectivity.""",
            
            'release_notes.md': """# Release Notes v2.3
- Added vector search to Help Center.
- Improved refund flow for subscription cancellations.
- Fixed bug in WhatsApp reconnection wizard."""
        }
        
        return content_map.get(filename, "")

class VectorIndex:
    """Manages embeddings and vector similarity search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding model and vector index."""
        
        print(f"🤖 Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None
        
    def create_embeddings(self, documents: List[str]) -> np.ndarray:
        """Create embeddings for all documents."""
        
        print(f"⚡ Creating embeddings for {len(documents)} chunks...")
        self.embeddings = self.encoder.encode(documents, show_progress_bar=True)
        return self.embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for similarity search."""
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"✅ Built FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Search for most similar documents to query."""
        
        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return distances[0], indices[0]

class QAAgent:
    """Main RAG QA Agent that orchestrates retrieval and generation."""
    
    def __init__(self):
        self.doc_store = DocumentStore()
        self.vector_index = VectorIndex()
        self.documents = []
        self.doc_metadata = []
        
        # Initialize QA model (using a lightweight model for efficiency)
        print("🧠 Loading QA model...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if torch.cuda.is_available() else -1
        )
        
    def setup(self):
        """Setup the RAG system by loading docs and creating index."""
        
        # Load documents
        self.doc_metadata = self.doc_store.load_documents()
        self.documents = [doc['content'] for doc in self.doc_metadata]
        
        # Create embeddings and build index
        embeddings = self.vector_index.create_embeddings(self.documents)
        self.vector_index.build_index(embeddings)
        
        print("✨ RAG system setup complete!")
    
    def answer_question(self, question: str, verbose: bool = True) -> Dict:
        """Answer a question using retrieval-augmented generation."""
        
        # Retrieve relevant documents
        distances, indices = self.vector_index.search(question, k=3)
        
        # Get the relevant chunks and their sources
        relevant_chunks = []
        sources = set()
        
        for idx in indices:
            chunk = self.documents[idx]
            source = self.doc_metadata[idx]['source']
            relevant_chunks.append(chunk)
            sources.add(source)
        
        # Combine context for QA
        context = "\n\n".join(relevant_chunks)
        
        # Generate answer using QA model
        try:
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=100
            )
            
            answer = result['answer']
            confidence = result['score']
        except Exception as e:
            # Fallback to simple extraction if QA model fails
            answer = self._extract_answer_fallback(question, context)
            confidence = 0.5
        
        # Format response
        response = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'sources': list(sources),
            'context_used': context[:500] + "..." if len(context) > 500 else context
        }
        
        if verbose:
            self._print_response(response)
        
        return response
    
    def _extract_answer_fallback(self, question: str, context: str) -> str:
        """Simple fallback answer extraction based on keywords."""
        
        # Convert to lowercase for matching
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Split context into sentences
        sentences = context.split('.')
        
        # Find most relevant sentence
        best_sentence = ""
        best_score = 0
        
        question_words = set(question_lower.split())
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Count matching words
            matches = sum(1 for word in question_words if word in sentence_lower)
            if matches > best_score:
                best_score = matches
                best_sentence = sentence.strip()
        
        return best_sentence if best_sentence else "Unable to find specific answer in documents."
    
    def _print_response(self, response: Dict):
        """Pretty print the response."""
        
        print(f"\n{'='*60}")
        print(f"❓ Question: {response['question']}")
        print(f"💡 Answer: {response['answer']}")
        print(f"📊 Confidence: {response['confidence']:.2%}")
        print(f"📁 Sources: {', '.join(response['sources'])}")
        print(f"{'='*60}\n")
    
    def evaluate_on_test_set(self, test_file: str = 'rag_eval_questions.csv'):
        """Evaluate the agent on the provided test questions."""
        
        print(f"\n🧪 Running evaluation on test questions...")
        
        # Load test questions
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
        else:
            # Use the provided test data
            df = pd.DataFrame({
                'qid': [1, 2, 3, 4, 5, 6],
                'question': [
                    'What is the return window for products purchased on WhatsApp?',
                    'How much discount can Service Plus get on annual billing?',
                    'Which Widget has IP67 and what Bluetooth version does it support?',
                    'If login fails, what should a user try first?',
                    'Name one improvement in Release Notes v2.3 related to refunds.',
                    'List the steps to onboard a new user up to installation.'
                ],
                'expected_answer_contains': [
                    '30 days',
                    '15%',
                    'Widget B, Bluetooth 5.3',
                    'reset the password',
                    'Improved refund flow',
                    'Create an account, Verify email and phone, Install'
                ],
                'source': [
                    'refund_policy.md',
                    'pricing.md',
                    'product_specs_widget.md',
                    'troubleshooting.md',
                    'release_notes.md',
                    'onboarding_guide.md'
                ]
            })
        
        results = []
        correct = 0
        
        for idx, row in df.iterrows():
            response = self.answer_question(row['question'], verbose=False)
            
            # Check if expected content is in answer
            expected_parts = row['expected_answer_contains'].split(',')
            is_correct = any(exp.strip().lower() in response['answer'].lower() 
                           for exp in expected_parts)
            
            if is_correct:
                correct += 1
            
            results.append({
                'qid': row['qid'],
                'question': row['question'],
                'expected': row['expected_answer_contains'],
                'actual_answer': response['answer'],
                'correct': is_correct,
                'confidence': response['confidence'],
                'sources_used': ', '.join(response['sources'])
            })
            
            print(f"Q{row['qid']}: {'✅' if is_correct else '❌'} - {row['question'][:50]}...")
        
        # Calculate accuracy
        accuracy = correct / len(df) * 100
        print(f"\n📈 Evaluation Results:")
        print(f"   Accuracy: {accuracy:.1f}% ({correct}/{len(df)} correct)")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('evaluation_results.csv', index=False)
        print(f"   Results saved to 'evaluation_results.csv'")
        
        return results_df, accuracy

def main():
    """Main execution function."""
    
    print("🚀 Starting RAG QA Agent...")
    print("-" * 60)
    
    # Initialize agent
    agent = QAAgent()
    
    # Setup (load docs and create index)
    agent.setup()
    
    # Test with a sample question
    print("\n📝 Testing with sample question:")
    sample_q = "What is the battery life of Widget B?"
    agent.answer_question(sample_q)
    
    # Run evaluation on test set
    results, accuracy = agent.evaluate_on_test_set()
    
    print("\n✅ RAG Agent ready for use!")
    print("You can now call: agent.answer_question('your question here')")
    
    return agent

if __name__ == "__main__":
    agent = main()