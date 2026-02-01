
from llm_chatbot import WebSearchTool

def test_web_search():
    print("Testing Web Search Tool...")
    tool = WebSearchTool()

    # Test valid search
    print("\n[Test 1] Searching for 'health benefits of kale'...")
    results = tool.search("health benefits of kale", max_results=1)
    print(f"Results:\n{results}")

    if "benefits" in results.lower() or "kale" in results.lower():
        print("✅ Test 1 Passed")
    else:
        print("❌ Test 1 Failed")

if __name__ == "__main__":
    test_web_search()
