import google.generativeai as genai
import json
import re
from typing import List, Dict, Any
from config import GEMINI_MODEL, GOOGLE_API_KEY

class QueryDecomposer:
    def __init__(self, api_key: str = None):
        genai.configure(api_key=api_key or GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def decompose_query(self, query: str) -> List[str]:
        print(f"\n🧩 Decomposing query into sub-queries: '{query}'")

        prompt = f"""
        You are an expert at breaking down complex questions into simpler, focused sub-questions.

        Given this query: "{query}"

        Break it down into 2-4 simpler, specific sub-questions that together would answer the original query.
        Each sub-question should:
        1. Be independently answerable
        2. Focus on one specific aspect
        3. Be clear and concise
        4. Together cover all aspects of the original query

        Return ONLY a JSON array of sub-questions, nothing else.
        Example format: ["What is X?", "How does Y work?", "What are the benefits of Z?"]

        Sub-questions:
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                sub_queries = json.loads(json_match.group())
                print(f"✅ Decomposed into {len(sub_queries)} sub-queries:")
                for i, sub_query in enumerate(sub_queries, 1):
                    print(f"   {i}. {sub_query}")
                return sub_queries
            else:
                print("❌ Failed to extract JSON from decomposition response")
                return [query]
        except Exception as e:
            print(f"❌ Error in query decomposition: {e}")
            return [query]

    def rerank_chunks(self, all_chunks: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        print(f"\n🔄 Reranking {len(all_chunks)} chunks for original query")

        seen_content = set()
        unique_chunks = []
        for chunk in all_chunks:
            content_hash = hash(chunk['content'])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)

        print(f"📊 Removed {len(all_chunks) - len(unique_chunks)} duplicate chunks")

        reranked_chunks = sorted(unique_chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)
        top_chunks = reranked_chunks[:8]
        print(f"📈 Selected top {len(top_chunks)} chunks after reranking")
        for i, chunk in enumerate(top_chunks, 1):
            score = chunk.get('similarity_score', 0)
            chunk_type = chunk.get('type', 'unknown')
            page = chunk.get('page_number', 'N/A')
            print(f"   {i}. Type: {chunk_type}, Page: {page}, Score: {score:.4f}")
        return top_chunks

    def combine_answers(self, original_query: str, sub_query_answers: List[Dict[str, Any]]) -> str:
        print(f"\n🔗 Combining {len(sub_query_answers)} sub-query answers")

        sub_answers_text = []
        for i, qa in enumerate(sub_query_answers, 1):
            sub_answers_text.append(f"Sub-question {i}: {qa['question']}\nAnswer: {qa['answer']}")

        combined_context = "\n\n".join(sub_answers_text)

        prompt = f"""
        You are tasked with combining multiple sub-answers into one comprehensive, coherent response.

        Original question: "{original_query}"

        Sub-question answers:
        {combined_context}

        Instructions:
        1. Synthesize the sub-answers into one comprehensive response
        2. Ensure the final answer directly addresses the original question
        3. Maintain consistency and avoid contradictions
        4. Include all relevant information from the sub-answers
        5. Present the information in a logical, well-structured manner
        6. If sub-answers complement each other, integrate them smoothly
        7. If there are any conflicts, note them appropriately

        Provide a comprehensive final answer:
        """

        try:
            response = self.model.generate_content(prompt)
            final_answer = response.text.strip()
            print("✅ Successfully combined sub-query answers into final response")
            print(f"📝 Final answer length: {len(final_answer)} characters")
            return final_answer
        except Exception as e:
            print(f"❌ Error combining answers: {e}")
            fallback_answer = f"Based on the analysis of your question '{original_query}':\n\n"
            for qa in sub_query_answers:
                fallback_answer += f"• {qa['question']}\n{qa['answer']}\n\n"
            return fallback_answer

    def log_query_flow(self, original_query: str, sub_queries: List[str],
                       sub_answers: List[Dict[str, Any]], final_answer: str):
        print(f"\n" + "=" * 80)
        print("📋 QUERY DECOMPOSITION FLOW SUMMARY")
        print("=" * 80)
        print(f"🎯 Original Query: {original_query}")
        print(f"🧩 Decomposed into {len(sub_queries)} sub-queries:")
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"   {i}. {sub_query}")
        print(f"\n📊 Sub-query Results:")
        for i, qa in enumerate(sub_answers, 1):
            print(f"   Sub-query {i}: {qa['question']}")
            print(f"   Answer length: {len(qa['answer'])} characters")
            print(f"   Chunks used: {qa.get('chunks_count', 'N/A')}")
        print(f"\n🎉 Final Answer length: {len(final_answer)} characters")
        print("=" * 80)
