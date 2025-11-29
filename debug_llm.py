#!/usr/bin/env python3
"""
LLM connection and response debugging script
"""

import os
from langchain_openai import ChatOpenAI
from simple_active_refine.knoweldge_retriever import RetrievedKnowledge
from simple_active_refine.util import get_logger
from settings import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
logger = get_logger('LLMDebug')

def test_basic_llm():
    """Test basic LLM functionality"""
    print("=== Basic LLM Test ===")
    
    try:
        llm = ChatOpenAI(model='gpt-4o', temperature=0)
        response = llm.invoke("Hello, can you respond with 'LLM is working'?")
        print(f"Basic LLM response: {response.content}")
        return True
    except Exception as e:
        print(f"Basic LLM test failed: {e}")
        return False

def test_structured_output():
    """Test structured output functionality"""
    print("\n=== Structured Output Test ===")
    
    try:
        llm = ChatOpenAI(model='gpt-4o', temperature=0)
        structured_llm = llm.with_structured_output(RetrievedKnowledge)
        
        prompt = """
Generate a simple knowledge retrieval result with 1 triple and 1 entity.
Return in JSON format:
{
    "triples": [{"subject": "london", "predicate": "capital_of", "object": "england", "source": "test"}],
    "entities": [{"id": "e1", "label": "England", "description_short": "A country", "description": null, "source": "test"}]
}
"""
        
        response = structured_llm.invoke(prompt)
        print(f"Structured response: {response}")
        print(f"Response type: {type(response)}")
        
        if response:
            print(f"Triples: {len(response.triples) if hasattr(response, 'triples') else 'No triples attr'}")
            print(f"Entities: {len(response.entities) if hasattr(response, 'entities') else 'No entities attr'}")
            return True
        else:
            print("Empty structured response")
            return False
            
    except Exception as e:
        print(f"Structured output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_search_llm():
    """Test LLM with web search tools"""
    print("\n=== Web Search LLM Test ===")
    
    try:
        # Try different web search configurations
        configs = [
            {"tools": [{"type": "web_search"}], "tool_choice": "auto"},
            {"tools": [{"type": "web_search_preview"}], "tool_choice": "auto"},
        ]
        
        for i, config in enumerate(configs):
            try:
                llm = ChatOpenAI(model='gpt-4o', temperature=0, **config)
                response = llm.invoke("What is the capital of France?")
                print(f"Web search config {i+1} response: {response.content[:100]}...")
                return True
            except Exception as e:
                print(f"Web search config {i+1} failed: {e}")
                continue
                
        print("All web search configurations failed")
        return False
        
    except Exception as e:
        print(f"Web search test failed: {e}")
        return False

def test_openai_api_key():
    """Test OpenAI API key validity"""
    print("\n=== API Key Test ===")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY found")
        return False
        
    print(f"API key present: {api_key[:10]}...{api_key[-4:]}")
    
    # Test with a simple OpenAI call
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API key works'"}],
            max_tokens=10
        )
        print(f"Direct OpenAI API response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Direct OpenAI API test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting LLM debugging...")
    
    # Run all tests
    tests = [
        ("OpenAI API Key", test_openai_api_key),
        ("Basic LLM", test_basic_llm),
        ("Structured Output", test_structured_output),
        ("Web Search LLM", test_web_search_llm),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        results[test_name] = test_func()
    
    print(f"\n{'='*50}")
    print("SUMMARY:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if not all(results.values()):
        print("\nSome tests failed. Check the issues above.")
    else:
        print("\nAll tests passed! LLM should be working correctly.")