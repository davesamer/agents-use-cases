import asyncio
from textwrap import dedent

from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.team.team import Team
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.hackernews import HackerNewsTools
from dotenv import load_dotenv

load_dotenv("configs/.env")

reddit_researcher = Agent(
    name="Reddit Researcher",
    role="Research a topic on Reddit",
    model=AzureOpenAI(id="gpt-35-turbo"),
    tools=[DuckDuckGoTools()],
    add_name_to_instructions=True,
    instructions=dedent("""
    You are a Reddit researcher.
    You will be given a topic to research on Reddit.
    You will need to find the most relevant posts on Reddit.
    """),
)

hackernews_researcher = Agent(
    name="HackerNews Researcher",
    model=AzureOpenAI("gpt-35-turbo"),
    role="Research a topic on HackerNews.",
    tools=[HackerNewsTools()],
    add_name_to_instructions=True,
    instructions=dedent("""
    You are a HackerNews researcher.
    You will be given a topic to research on HackerNews.
    You will need to find the most relevant posts on HackerNews.
    """),
)

academic_paper_researcher = Agent(
    name="Academic Paper Researcher",
    model=AzureOpenAI("gpt-35-turbo"),
    role="Research academic papers and scholarly content",
    tools=[GoogleSearchTools(), ArxivTools()],
    add_name_to_instructions=True,
    instructions=dedent("""
    You are a academic paper researcher.
    You will be given a topic to research in academic literature.
    You will need to find relevant scholarly articles, papers, and academic discussions.
    Focus on peer-reviewed content and citations from reputable sources.
    Provide brief summaries of key findings and methodologies.
    """),
)

twitter_researcher = Agent(
    name="Twitter Researcher",
    model=AzureOpenAI("gpt-35-turbo"),
    role="Research trending discussions and real-time updates",
    tools=[DuckDuckGoTools()],
    add_name_to_instructions=True,
    instructions=dedent("""
    You are a Twitter/X researcher.
    You will be given a topic to research on Twitter/X.
    You will need to find trending discussions, influential voices, and real-time updates.
    Focus on verified accounts and credible sources when possible.
    Track relevant hashtags and ongoing conversations.
    """),
)


agent_team = Team(
    name="Discussion Team",
    mode="collaborate",
    model=AzureOpenAI("gpt-35-turbo"),
    members=[
        reddit_researcher,
        hackernews_researcher,
        academic_paper_researcher,
        twitter_researcher,
    ],
    instructions=[
        "You are a discussion master.",
        "You have to stop the discussion when you think the team has reached a consensus.",
    ],
    success_criteria="The team has reached a consensus.",
    enable_agentic_context=True,
    show_tool_calls=True,
    markdown=True,
    show_members_responses=True,
)

if __name__ == "__main__":

    # "Start the discussion on the topic: What is a good strategy to educate the employees of our company about AI usage?"

    # asyncio.run(
    #     agent_team.print_response(
    #         message="How to start an AI community to share ideas & best practices and foster collaboration with members from different organizations",
    #         stream=True,
    #         stream_intermediate_steps=True,
    #     )
    # )

    asyncio.run(
        agent_team.print_response(
            message="What are very valuable and concrete AI Agent use cases for regional banks in the DACH area?",
            stream=True,
            stream_intermediate_steps=True,
        )
    )


