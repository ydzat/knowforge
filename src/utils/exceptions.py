"""
异常处理类模块，包含项目中所有自定义异常类
"""


class NoteGenError(Exception):
    """基础异常：所有可预期错误继承它"""
    pass


class ConfigError(NoteGenError):
    """配置相关错误"""
    pass


class InputError(NoteGenError):
    """输入处理相关错误"""
    pass


class APIError(NoteGenError):
    """API调用相关错误"""
    pass


class MemoryError(NoteGenError):
    """向量记忆相关错误"""
    pass


class OutputError(NoteGenError):
    """输出生成相关错误"""
    pass


class LocaleError(NoteGenError):
    """多语言支持相关错误"""
    pass