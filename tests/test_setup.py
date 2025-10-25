import os

import boto3
from dotenv import load_dotenv

load_dotenv()


def test_aws_connection():
    """Test AWS connection and Bedrock access"""
    print("Testing AWS Connection...")

    try:
        # Test STS
        sts = boto3.client("sts", region_name="us-west-2")
        identity = sts.get_caller_identity()
        print(f"✅ AWS Identity: {identity['Arn']}")

        # Test Bedrock
        bedrock = boto3.client("bedrock", region_name="us-west-2")
        models = bedrock.list_foundation_models()

        claude_models = [
            m for m in models["modelSummaries"] if "claude-3-haiku" in m["modelId"]
        ]

        if claude_models:
            print(f"✅ Claude 3 Haiku available: {claude_models[0]['modelId']}")
        else:
            print("❌ Claude 3 Haiku not found - request access in AWS Console")

        return True

    except Exception as e:
        print(f"❌ AWS Error: {e}")
        return False


def test_news_api():
    """Test News API connection"""
    print("\nTesting News API...")

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key or api_key == "your-news-api-key-here":
        print("❌ News API key not set in .env file")
        return False

    try:
        from newsapi import NewsApiClient

        client = NewsApiClient(api_key=api_key)

        # Test API call
        response = client.get_everything(q="battery", page_size=1)
        print(f"✅ News API working: {len(response['articles'])} articles found")
        return True

    except Exception as e:
        print(f"❌ News API Error: {e}")
        return False


if __name__ == "__main__":
    print("BATTERY ANALYST AGENTS - AWS SETUP TEST")
    print("=" * 50)

    aws_ok = test_aws_connection()
    news_ok = test_news_api()

    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    print(f"AWS Bedrock: {'✅ Ready' if aws_ok else '❌ Not Ready'}")
    print(f"News API: {'✅ Ready' if news_ok else '❌ Not Ready'}")

    if aws_ok and news_ok:
        print("\nAll systems ready! You can now run the agents.")
    else:
        print("\nPlease fix the issues above before running agents.")
