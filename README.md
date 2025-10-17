## Gemini Image Plugin (gcli2api)

- 标识名：`gemini-image`
- 功能：通过 gcli2api 转发 Gemini 生图/改图，自动发送到 QQ（Napcat）。
- 设计：模块化/可扩展，解耦 API 客户端，与 AstrBot 交互通过 `llm_tool` 与指令组。

### 安装与配置

1. 将本目录放入 AstrBot 插件目录。
2. 在 AstrBot 后台或配置文件中为该插件配置：

```json
{
  "gcli2api_base_url": "http://127.0.0.1:7861",
  "gcli2api_api_password": "pwd",
  "model_name": "gemini-2.5-flash-image",
  "max_retry_attempts": 3,
  "nap_server_address": "",
  "nap_server_port": 0
}
```

### 指令

- `/生图 <提示词>`：纯文本生图。
- `/改图 <提示词>`：基于消息中携带/引用的图片进行改图。
- `/手办化`：携带/引用图片后，使用内置提示词进行“手办化”改图。
- `/手办化2`：携带/引用图片后，使用更严格的内置规则进行“手办化”改图。
- `/aiimg帮助`：查看用法说明。
  （配置修改请在 AstrBot 插件设置中进行，无需命令）

### 发送到 QQ

- 优先使用 `callback_api_base`（AstrBot 全局配置）生成临时下载链接；失败则回退到本地文件发送。
- 如配置了 Napcat 文件中转（`nap_server_address/port`），将先上传文件以便外部访问。

### 与 gcli2api 的对接

- 端点：`/v1beta/models/{model}:generateContent`（非流式），`/v1beta/models/{model}:streamGenerateContent`（流式，默认附加 `?alt=sse`）。
- 鉴权：若配置了 `gcli2api_api_password`，将使用请求头 `x-goog-api-key: <password>`；也可通过 URL `?key=`（插件默认用请求头）。
- 负载：`contents=[{role:user, parts:[{text}, {inlineData}...]}]`，与官方 SDK 示例一致；改图时将用户图片转为 inlineData 放入 parts。

### 设计说明

- 遵循 AstrBot 插件规范：`metadata.yaml` + `@register` + `filter.command`。
- 扩展性：`utils/gemini_images_api.py` 封装 API 调用，端点与模型可配置。
- 解耦：业务逻辑与网络请求分离，专注 gcli2api 转发与 AstrBot 交互。
- 开闭原则：新增模型/路径仅需修改配置或替换 API 客户端，无需改动指令/对外接口。

### 注意事项

- 请确保 gcli2api 已配置可用的 Google 凭证；插件侧仅负责负载与鉴权的适配。
