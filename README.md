

# 💱 Steam游戏评论爬取+简易分析

可以爬取steam游戏评论，并进行简易分析，如有新功能添加需求，自行添加即可。

*ps：起因是想想看看《丝之歌》究竟怎么样~*

![项目Logo](image/main.png)




## 🚀 快速使用 Quick Start

**方法：于终端运行文件**
*普通用户使用*

环境配置：
```
pip install requests pandas pyarrow tqdm
pip install streamlit pandas pyarrow plotly
```

# 1) 输入游戏appid（默认是丝之歌的1030300）
双击打开python fetch_and_dump.py文件，翻到最下面，按照注释修改即可。

参数修改：一页是100条评论，max_pages_per_app为600，就是6w条评论，看你需求，越大时间越长。

# 2) 抓数据（示例先 600 页/APP）
打开终端，输入
```
python fetch_and_dump.py
```
# 3) 打开看板
打开终端，输入
```
streamlit run app.py
```


# 📬 联系我

感谢你对 **Steam游戏评论爬取+简易分析** 项目的关注与支持！如果你在使用过程中有任何问题、建议，欢迎通过以下方式联系我：

---

## 📧 邮箱

> q782428471@gmail.com

## 💼 GitHub

> [GitHub Profile](https://github.com/MiracleGodForlove/)
