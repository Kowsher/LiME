import torch
from torch.utils.data import Dataset
from PIL import Image
import av
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import os
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor


def read_video_pyav(container, indices):
    """Read specific frames from video using PyAV."""
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_video_frames(video_path: str, num_frames: int = 8):
    """Load video and sample frames uniformly."""
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(video=0))
        container.seek(0)
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    video_frames = read_video_pyav(container, indices)
    container.close()
    
    return [Image.fromarray(frame) for frame in video_frames]


def find_assistant_response_start(input_ids: torch.Tensor, tokenizer) -> int:
    """
    Find the position where assistant's response starts.
    Returns the index of the first token AFTER '<|im_start|>assistant\n'
    """
    # Get the token IDs for the assistant marker
    # For Qwen2-based models: <|im_start|>assistant\n
    assistant_marker = "<|im_start|>assistant\n"
    marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)
    
    input_ids_list = input_ids.tolist()
    marker_len = len(marker_ids)
    
    # Find the LAST occurrence of assistant marker (in case of multi-turn)
    last_pos = -1
    for i in range(len(input_ids_list) - marker_len + 1):
        if input_ids_list[i:i + marker_len] == marker_ids:
            last_pos = i + marker_len  # Position AFTER the marker
    
    return last_pos


class MultiModalDataset(Dataset):
    """Dataset that handles text, image, and video samples."""
    
    def __init__(
        self, 
        dataset,
        processor: LlavaOnevisionProcessor,
        data_root: str = "",
        num_video_frames: int = 8,
        max_length: int = 2048,
    ):
        self.dataset = dataset
        self.processor = processor
        self.data_root = data_root
        self.num_video_frames = num_video_frames
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        source_type = item['source_type']
        media_path = item['media_path']
        question = item['question']
        answer = item['answer']
        
        if source_type == 'text':
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            images = None
            videos = None
            
        elif source_type == 'image':
            conversation = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            image_path = os.path.join(self.data_root, media_path)
            images = [Image.open(image_path).convert('RGB')]
            videos = None
            
        elif source_type == 'video':
            conversation = [
                {"role": "user", "content": [
                    {"type": "video"},
                    {"type": "text", "text": question}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            video_path = os.path.join(self.data_root, media_path)
            videos = [load_video_frames(video_path, self.num_video_frames)]
            images = None
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
        
        # Process fully here
        processed = self.processor(
            text=prompt,
            images=images,
            videos=videos,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        result = {}
        for k, v in processed.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.squeeze(0)
            else:
                result[k] = v
        
        # ⭐ Find where assistant response starts and store it
        assistant_start = find_assistant_response_start(
            result['input_ids'], 
            self.processor.tokenizer
        )
        result['assistant_start_idx'] = assistant_start
        result['source_type'] = source_type
        
        return result


@dataclass
class MultiModalCollator:
    """Collator that pads pre-processed tensors and masks labels properly."""
    
    processor: LlavaOnevisionProcessor
    max_length: int = 2048
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        
        max_len = min(
            max(item['input_ids'].shape[0] for item in batch),
            self.max_length
        )
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        all_pixel_values = []
        all_image_sizes = []
        all_pixel_values_videos = []
        
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        video_token_id = self.processor.tokenizer.convert_tokens_to_ids("<video>")
        im_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        for item in batch:
            input_ids = item['input_ids']
            seq_len = input_ids.shape[0]
            
            if seq_len > max_len:
                input_ids = input_ids[:max_len]
                seq_len = max_len
            
            padding_length = max_len - seq_len
            if padding_length > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            else:
                attention_mask = torch.ones(seq_len, dtype=torch.long)
            
            # ⭐ Create labels
            labels = input_ids.clone()
            
            # ⭐ Find the SECOND <|im_start|> (assistant's turn)
            im_start_positions = (input_ids == im_start_token_id).nonzero(as_tuple=True)[0]
            
            if len(im_start_positions) >= 2:
                # Mask everything before the second <|im_start|>
                second_im_start = im_start_positions[1].item()
                labels[:second_im_start] = -100
            
            # Mask padding tokens
            labels[input_ids == pad_token_id] = -100
            
            # Mask image/video tokens
            if image_token_id != self.processor.tokenizer.unk_token_id:
                labels[input_ids == image_token_id] = -100
            if video_token_id != self.processor.tokenizer.unk_token_id:
                labels[input_ids == video_token_id] = -100
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            
            if 'pixel_values' in item:
                pv = item['pixel_values']
                if pv.dim() == 3:
                    pv = pv.unsqueeze(0)
                all_pixel_values.append(pv)
                
                if 'image_sizes' in item:
                    img_sizes = item['image_sizes']
                    if img_sizes.dim() == 1:
                        img_sizes = img_sizes.unsqueeze(0)
                    all_image_sizes.append(img_sizes)
            
            if 'pixel_values_videos' in item:
                pv_vid = item['pixel_values_videos']
                if pv_vid.dim() == 4:
                    pv_vid = pv_vid.unsqueeze(0)
                all_pixel_values_videos.append(pv_vid)
        
        result = {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'labels': torch.stack(batch_labels),
        }
        
        if all_pixel_values:
            result['pixel_values'] = torch.cat(all_pixel_values, dim=0)
            if all_image_sizes:
                result['image_sizes'] = torch.cat(all_image_sizes, dim=0)
        
        if all_pixel_values_videos:
            result['pixel_values_videos'] = torch.cat(all_pixel_values_videos, dim=0)
        
        return result
