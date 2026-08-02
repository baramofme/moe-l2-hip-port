"""GGUF metadata 内嵌映射（P2，2026-08-02）。

把 domain_expert_map.json 内嵌进 GGUF 模型文件的自定义 metadata key，
作为可选发布形态（方便分发，不依赖 ~/.moe-l2/maps/ 固定位置）。

关键点：
- GGUF 格式的 metadata KV 区在文件头部、tensor data 在末尾，
  新增 KV 必须**全文件重建**（读全部 tensor → 写新文件）。
  因此本模块定位为"发布时做一次"（embed），不是运行时操作。
- 自定义 key：`moe_l2.domain_expert_map`（STRING，JSON 文本）
- 读取时 GGUFReader 直接读 KV，无需重建。

用法：
    from moe_l2.gguf_embed import embed_map, read_embedded_map
    embed_map("model.gguf", "domain_expert_map.json", "model.embedded.gguf")
    mapping = read_embedded_map("model.embedded.gguf")  # dict or None
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from gguf import GGUFReader, GGUFWriter

logger = logging.getLogger("moe-l2-gguf-embed")

# 自定义 metadata key（GGUF 惯例：命名空间前缀）
EMBED_KEY = "moe_l2.domain_expert_map"


def embed_map(
    model_path: str | Path,
    map_path: str | Path,
    output_path: str | Path,
    keep_original: bool = False,
) -> Path:
    """把 domain_expert_map.json 内嵌进模型文件（全文件重建）。

    Args:
        model_path: 源 GGUF 模型。
        map_path: domain_expert_map.json 路径。
        output_path: 输出文件（建议 .embedded.gguf 后缀）。
        keep_original: True 保留源文件；False 时重建成功后删除源文件。

    Returns:
        输出文件路径。

    Raises:
        ValueError: 模型已含 EMBED_KEY（避免重复内嵌）。
    """
    model_path = Path(model_path)
    map_path = Path(map_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")

    with open(map_path, encoding="utf-8") as f:
        mapping = json.load(f)
    map_text = json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))

    reader = GGUFReader(str(model_path))

    # 防重复内嵌
    if EMBED_KEY in reader.fields:
        raise ValueError(
            f"Model already contains {EMBED_KEY} — refusing to double-embed"
        )

    # 幂等输出：先写临时文件，成功后 rename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        # architecture 字段：parts[-1] 是 uint8 memmap → bytes → decode
        arch_raw = bytes(reader.fields["general.architecture"].parts[-1])
        arch = arch_raw.decode("utf-8", "replace").strip("\x00")
        writer = GGUFWriter(str(tmp_path), arch=arch, endianess=reader.endianess)

        # 对齐值保持一致（llama.cpp 依赖对齐做 tensor 偏移）
        align_field = reader.fields.get("general.alignment")
        if align_field is not None:
            writer.data_alignment = int(align_field.parts[-1].item())
        logger.info("data_alignment=%d", writer.data_alignment)

        # 1. 复制原文件全部 KV metadata（跳过系统 KV——
        #    general.architecture 由 add_architecture 自动写；
        #    GGUF.version/tensor_count/kv_count 由 write_header 自动 pack）
        for key, field in reader.fields.items():
            if key == "general.architecture" or key.startswith("GGUF."):
                continue
            if key == "moe_l2.domain_expert_map":
                continue  # 防重复内嵌（上面已查过，防御性跳过）
            _copy_field_auto(writer, key, field)

        # 2. 追加自定义映射
        writer.add_string(EMBED_KEY, map_text)

        # 3. 复制全部 tensor info（官方：data.shape + data.nbytes + tensor_type）
        for tensor in reader.tensors:
            writer.add_tensor_info(
                tensor.name,
                [int(d) for d in tensor.data.shape],
                tensor.data.dtype,
                tensor.data.nbytes,
                tensor.tensor_type,
            )

        # 4. 写 header + KV
        writer.write_header_to_file()
        writer.write_kv_data_to_file()

        # 5. 写 tensor info 表（写完后 state=TI_DATA，write_tensor_data 才能调用）
        writer.write_ti_data_to_file()

        # 6. 逐个复制 tensor data（memmap 直接引用原数据 → 写新文件；
        #    显式传原文件 endianess，避免字节序差异）
        for tensor in reader.tensors:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)

        # 清理 writer 内部缓冲
        writer.close()
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # 输出落位
    if output_path.exists():
        output_path.unlink()
    tmp_path.rename(output_path)

    # 可选：删除源文件
    if not keep_original and model_path != output_path:
        model_path.unlink(missing_ok=True)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(
        "Embedded %s (%d bytes JSON) → %s (%.1f MB)",
        EMBED_KEY,
        len(map_text),
        output_path,
        size_mb,
    )
    return output_path


def read_embedded_map(model_path: str | Path) -> dict | None:
    """从 GGUF 读取内嵌的 domain_expert_map（无则返回 None）。"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    reader = GGUFReader(str(model_path))
    field = reader.fields.get(EMBED_KEY)
    if field is None:
        return None
    raw = field.parts[-1]
    text = bytes(raw).decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Corrupt embedded map in %s: %s", model_path, e)
        return None


def _copy_field_auto(writer: GGUFWriter, key: str, field) -> None:
    """按官方 gguf_new_metadata 方式复制 KV：field.contents() + add_key_value。

    相比手动解析 parts，contents() 由 gguf 库统一处理标量/字符串/数组，
    类型和子类型也由 field.types 直接给出，避免类型误判。
    """
    from gguf.constants import GGUFValueType

    val_type = field.types[0] if field.types else GGUFValueType.STRING
    sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY and len(field.types) > 1 else None
    try:
        value = field.contents()
    except Exception as e:
        logger.warning("Skipping %s (contents() failed: %s)", key, e)
        return
    if value is None:
        return
    try:
        writer.add_key_value(key, value, val_type, sub_type=sub_type)
    except Exception as e:
        logger.warning("Skipping %s (add_key_value failed: %s)", key, e)
