# Streaming Interaction 标注规范

## 目标

Streaming Interaction 数据用于训练模型在流式场景中结合当前时间点画面和历史对话给出下一步建议。

典型交互形式：

```text
用户：我现在应该怎么做？
AI：根据当前画面，你处在起始阶段，接下来...
用户：我已经做到这一步了，下一步呢？
AI：结合上一轮和当前画面，继续...
```

## 输入要求

该 pipeline 的输入可以是任意 JSON list，只要每条样本包含：

- `id`
- `video`
- `metadata.task_id` / `metadata.task_name`
- `metadata.temporal_segments`

当前配置用 `temporal_caption_v001` 作为示例源，但不要求上游数据必须叫 v001/v002。

从输入样本生成流式交互时：

- 每个 `temporal_segment` 取中点作为一个交互时间节点。
- 每条样本最多生成 3 轮。
- 每一轮保留 `time_sec`、`frame_index`、`segment_id` 和历史对话。

## Canonical 格式

```json
{
  "id": "sample_stream",
  "video": "/path/to/video.mkv",
  "metadata": {
    "task_id": "seq_flex",
    "task_name": "序列收回",
    "interaction_type": "time_node_guidance",
    "review_status": "streaming_unreviewed"
  },
  "turns": [
    {
      "turn_id": "turn_000",
      "time_sec": 3.75,
      "frame_index": 225,
      "segment_id": "seg_000",
      "history": [],
      "user": "我现在应该怎么做？",
      "assistant": "从当前时间点看..."
    }
  ]
}
```

## Messages 格式

训练入口如果只支持多轮 messages，可以使用：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<video time=3.75s frame=225>\n我现在应该怎么做？"
    },
    {
      "role": "assistant",
      "content": "从当前时间点看..."
    }
  ]
}
```

## 复核重点

人工复核时重点检查：

1. `time_sec` 对应的画面是否落在正确阶段。
2. assistant 是否同时利用当前画面和历史对话。
3. 下一步建议是否越过当前阶段，或者和任务无关。
4. 如果源 temporal segment 不是 `verified`，不要把回答写成精确边界事实。
5. 流式问答要像真实助手，不要只复述标签。

## 限制

当前第一版是从 temporal segment seed 自动生成的，适合训练“时间节点 + 历史对话”的格式，不等于真实应用场景中的售票机、导航或复杂环境问答。迁移到真实场景时，可以复用该输入/输出格式，但需要新的场景数据、时间节点和人工标注。
