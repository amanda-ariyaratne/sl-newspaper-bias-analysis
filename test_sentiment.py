#!/usr/bin/env python3
"""
Simple test script for sentiment analysis functionality.

This tests the core sentiment analysis components on sample data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.sentiment import analyze_sentiment, estimate_llm_cost, get_sentiment_stats


def test_local_sentiment():
    """Test local sentiment analyzer."""
    print("\n" + "="*70)
    print("Testing Local Sentiment Analyzer")
    print("="*70)

    # Sample articles
    articles = [
        {
            "id": "test-1",
            "title": "Economic Growth Soars to New Heights",
            "content": "The economy showed remarkable growth this quarter, exceeding all expectations with record-breaking performance."
        },
        {
            "id": "test-2",
            "title": "Devastating Floods Affect Thousands",
            "content": "Severe flooding has caused widespread destruction, displacing thousands of families and causing significant damage."
        },
        {
            "id": "test-3",
            "title": "Government Announces New Policy",
            "content": "Officials presented the new policy framework during a press conference yesterday."
        }
    ]

    try:
        results = analyze_sentiment(articles, model_type="local", show_progress=False)

        print(f"\nAnalyzed {len(results)} articles")
        for result in results:
            print(f"\nArticle: {articles[int(result.article_id.split('-')[1])-1]['title']}")
            print(f"  Overall Sentiment: {result.overall_sentiment:.2f}")
            print(f"  Headline Sentiment: {result.headline_sentiment:.2f}")
            print(f"  Confidence: {result.overall_confidence:.2f}")

        stats = get_sentiment_stats(results)
        print(f"\nStatistics:")
        print(f"  Average sentiment: {stats['avg_sentiment']:.2f}")
        print(f"  Sentiment range: [{stats['min_sentiment']:.2f}, {stats['max_sentiment']:.2f}]")

        print("\n✓ Local sentiment analysis test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Local sentiment analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cost_estimation():
    """Test cost estimation for LLM."""
    print("\n" + "="*70)
    print("Testing LLM Cost Estimation")
    print("="*70)

    try:
        cost = estimate_llm_cost(100)
        print(f"\nEstimated cost for 100 articles:")
        print(f"  Model: {cost['model']}")
        print(f"  Input tokens: {cost['total_input_tokens']:,}")
        print(f"  Output tokens: {cost['total_output_tokens']:,}")
        print(f"  Total cost: ${cost['total_cost_usd']:.2f}")

        assert cost['total_cost_usd'] > 0, "Cost should be positive"
        print("\n✓ Cost estimation test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Cost estimation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_result_conversion():
    """Test SentimentResult to dict conversion."""
    print("\n" + "="*70)
    print("Testing SentimentResult Conversion")
    print("="*70)

    from src.sentiment import SentimentResult

    try:
        result = SentimentResult(
            article_id="test-123",
            model_type="local",
            model_name="test-model",
            overall_sentiment=2.5,
            overall_confidence=0.85,
            headline_sentiment=3.0,
            headline_confidence=0.90,
            reasoning="Test reasoning",
            aspects={"tone": "positive"},
            processing_time_ms=100
        )

        result_dict = result.to_dict()

        assert result_dict["article_id"] == "test-123"
        assert result_dict["overall_sentiment"] == 2.5
        assert result_dict["sentiment_reasoning"] == "Test reasoning"

        print("\nSentimentResult fields:")
        for key, value in result_dict.items():
            print(f"  {key}: {value}")

        print("\n✓ SentimentResult conversion test passed!")
        return True

    except Exception as e:
        print(f"\n❌ SentimentResult conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS TEST SUITE")
    print("="*70)

    tests = [
        ("SentimentResult Conversion", test_sentiment_result_conversion),
        ("Cost Estimation", test_cost_estimation),
        ("Local Sentiment Analysis", test_local_sentiment)
    ]

    results = []
    for test_name, test_func in tests:
        passed = test_func()
        results.append((test_name, passed))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nPassed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} test(s) failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
