import re
from pathlib import Path
import pandas as pd

def extract_log_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 提取第二行参数
    args_line = lines[1].strip()
    args_dict = {}
    
    # 解析Namespace参数
    if args_line.startswith('Namespace('):
        args_str = args_line[10:-1]  # 去掉"Namespace("和结尾的")"
        pairs = args_str.split(', ')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                # 尝试转换数字类型
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # 保持字符串类型，去掉可能的引号
                    value = value.strip("'\"")
                args_dict[key] = value
    
    # 提取最后一行mse和mae
    last_line = lines[-1].strip()
    mse_match = re.search(r'mse:([\d.]+)', last_line)
    mae_match = re.search(r'mae:([\d.]+)', last_line)
    
    result_dict = {}
    if mse_match:
        result_dict['mse'] = float(mse_match.group(1))
    if mae_match:
        result_dict['mae'] = float(mae_match.group(1))
    
    return args_dict, result_dict

def get_filenames(directory, keyword):
    path = Path(directory)
    # 使用 rglob 进行递归匹配，pattern 中使用 '*' 表示通配
    # 例如，模式 f'*{keyword}*' 会匹配所有包含 keyword 的文件名
    found_files = list(path.rglob(f'*{keyword}*'))
    return found_files

if __name__ == "__main__":
    keyword = "25_29_13_28_PCLE_v4_ETTh1"
    directory = "./PatchTST_supervised/logs/LongForecasting/"
    attention_args = ['pred_len', 'fc_dropout', 'head_dropout', 'dropout', 'd_model', 'd_ff', 'pcle_out_dims', 'pcle_hidden_dims', 'pcle_proj_hidden_dims', 'random_seed', 'mse', 'mae']

    files = get_filenames(directory, keyword)

    # 创建DataFrame
    df_data = []
    for file_path in files:
        args_dict, result_dict = extract_log_data(file_path)
        
        # 创建一行数据
        row_data = {}

        # 添加文件名信息
        row_data['file_name'] = file_path.name
        
        # 提取attention_args中指定的参数
        for param in attention_args:
            if param in ['mse', 'mae']:
                # 从result_dict中获取mse和mae
                row_data[param] = result_dict.get(param, None)
            else:
                # 从args_dict中获取其他参数
                row_data[param] = args_dict.get(param, None)
        
        df_data.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(df_data)

    # 确保输出目录存在
    output_dir = Path("./exp_results")
    output_dir.mkdir(exist_ok=True)
    
    # 保存为Excel文件（xlsx格式）
    output_path = output_dir / f"{keyword}.xlsx"
    df.to_excel(output_path, index=False)
    
    print(f"数据已保存到: {output_path}")
    print(f"共处理了 {len(df_data)} 个文件")
    print(f"DataFrame形状: {df.shape}")