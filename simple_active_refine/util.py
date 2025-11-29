import logging

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    名前付きロガーを生成して返す関数

    Parameters
    ----------
    name : str
        ロガー名（通常は __name__ を渡す）
    level : int
        ログレベル (logging.DEBUG, logging.INFO, ...)

    Returns
    -------
    logging.Logger
        設定済みのロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 二重追加を防ぐ（同じロガーに複数ハンドラがつくのを回避）
    if not logger.handlers:
        # コンソール出力用ハンドラ
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # フォーマッタ設定
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # ロガーにハンドラを追加
        logger.addHandler(handler)

    return logger
