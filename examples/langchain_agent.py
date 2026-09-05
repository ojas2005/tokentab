"""Budget-capping a LangChain chain or agent.

    pip install 'tokentab[langchain]'
    python examples/langchain_agent.py

Needs langchain-core. The model call itself is faked, so no API key is required.
"""

from __future__ import annotations

from uuid import uuid4

from tokentab import BudgetExceededError, CostTracker, TokenTabCallbackHandler


def main() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    # In real use you never call these by hand -- LangChain fires them for you:
    #
    #     handler = TokenTabCallbackHandler(budget=5.00, tag="research-agent")
    #     agent.invoke({"input": "..."}, config={"callbacks": [handler]})
    #
    # This example drives the callbacks directly so it can run offline.

    handler = TokenTabCallbackHandler(budget=0.05, tag="research-agent")
    serialized = {"id": ["langchain", "chat_models", "ChatAnthropic"], "name": "ChatAnthropic"}

    for step in range(1, 6):
        run_id = uuid4()
        try:
            handler.on_chat_model_start(
                serialized,
                [[HumanMessage(content=f"Step {step}: research the topic in depth. " * 200)]],
                run_id=run_id,
                # max_tokens lets the guard budget for the response, not just
                # the prompt -- without it the session can overshoot by a call.
                invocation_params={"model": "claude-sonnet-4-5", "max_tokens": 1000},
            )
        except BudgetExceededError as exc:
            print(f"step {step}: STOPPED before calling the model -- {exc.limit_type} limit "
                  f"(${exc.projected:.4f} > ${exc.limit:.4f})")
            break

        handler.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(
                content="findings...",
                usage_metadata={"input_tokens": 2600, "output_tokens": 900,
                                "total_tokens": 3500},
            ))]]),
            run_id=run_id,
        )
        print(f"step {step}: ok, running total ${handler.total_cost:.4f}")

    print()
    print(handler.report().summary())

    print()
    print("Sharing one budget across a chain and your own code:")
    with CostTracker(budget=10.00, per_day=100.00, name="run") as run:
        shared = TokenTabCallbackHandler()   # no budget of its own: joins `run`
        run_id = uuid4()
        shared.on_chat_model_start(
            serialized, [[HumanMessage(content="hello")]], run_id=run_id,
            invocation_params={"model": "claude-sonnet-4-5"},
        )
        shared.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(
                content="hi", usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                              "total_tokens": 15},
            ))]]),
            run_id=run_id,
        )
        print(f"  the enclosing tracker sees {len(run.records)} call(s), "
              f"${run.total_cost:.6f}")


if __name__ == "__main__":
    main()
