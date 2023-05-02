import os


def get_boundary(config):
    return '-' * (sum(config.LOG_COL_LEN) + len(config.LOG_COL) + 1) + '\n'


def get_elements(item: str, length: int):
    """
    得到一个单元格的字符内容, 默认居中
    item: 要写的内容
    length: 这个单元格的长度
    """
    if len(item) < length:
        ele = ''
        space_len = (length - len(item)) // 2
        ele += ' ' * space_len
        ele += item
        ele += ' ' * (length - space_len - len(item))
    else:
        ele = item
    return ele


def get_head(config):
    boundary = get_boundary(config)
    line = '|'
    for i in range(len(config.LOG_COL)):
        line += get_elements(config.LOG_COL[i], config.LOG_COL_LEN[i])
        line += '|'
    line += '\n'
    out = boundary + line + boundary
    return out


def update_log(config, contents: dict):
    out = 'Outputs path: ' + config.OUTPUTS_PATH + '\n'
    out += get_head(config)
    rows = max([len(x) for x in contents.values()])
    for i in range(rows):
        line = '|'
        for j in range(len(config.LOG_COL)):
            head = config.LOG_COL[j]
            if len(contents[head]) > i:
                line += get_elements(contents[head][i], config.LOG_COL_LEN[j])
            else:
                line += ' ' * config.LOG_COL_LEN[j]
            line += '|'
        out += (line + '\n')
    boundary = get_boundary(config)
    out += boundary
    return out


def write_log(config, contents: str):
    if not os.path.exists(os.path.dirname(config.LOG_FILE)):
        os.makedirs(os.path.dirname(config.LOG_FILE))
    with open(config.LOG_FILE, 'a+') as f:
        f.write(contents)
    f.close()
    return 