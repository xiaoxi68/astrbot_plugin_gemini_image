from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.all import *
from astrbot.core.message.components import Reply

from .utils.gemini_images_api import generate_or_edit_image_gemini
from .utils.file_send_server import send_file


@register("gemini-image", "Codex", "对接 gcli2api 的 Gemini 生图/改图并发送到 QQ", "0.3.0")
class GeminiImagePlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        # 仅 gcli2api 后端
        default_base = (config.get("gcli2api_base_url") or "http://127.0.0.1:7861").strip()
        # 固定端点（强制 v1beta），不再提供配置项
        self.api_base = default_base
        self._GEN_PATH = "/v1beta/models/{model}:generateContent"
        self._STREAM_GEN_PATH = "/v1beta/models/{model}:streamGenerateContent"

        # 模型与重试
        self.model_name = (config.get("model_name") or "gemini-2.5-flash-image").strip()
        self.max_retry_attempts = int(config.get("max_retry_attempts", 3))
        # 固定策略：默认启用流式，附带 alt=sse；不提供可配置项
        self.use_stream = True

        # gcli2api 鉴权（默认 pwd）
        self.gcli2api_api_password = (config.get("gcli2api_api_password") or "pwd").strip()

        # Napcat 文件转发（可选）
        self.nap_server_address = config.get("nap_server_address")
        self.nap_server_port = config.get("nap_server_port")

        self._global_config_loaded = False

    async def _load_global_config(self):
        if self._global_config_loaded:
            return
        try:
            plugin_config = await sp.global_get("gemini-image", {})
            if "gcli2api_base_url" in plugin_config:
                self.api_base = str(plugin_config["gcli2api_base_url"]).strip() or self.api_base
                logger.info(f"从全局配置加载 gcli2api_base_url: {self.api_base}")
            if "model_name" in plugin_config:
                self.model_name = str(plugin_config["model_name"]).strip() or self.model_name
                logger.info(f"从全局配置加载 model_name: {self.model_name}")
            # 不再加载端点与流式相关配置项（固定策略）
            if "gcli2api_api_password" in plugin_config:
                self.gcli2api_api_password = str(plugin_config["gcli2api_api_password"]).strip() or self.gcli2api_api_password
            # 不再加载生成参数与系统指令（固定策略）
        except Exception as e:
            logger.error(f"加载全局配置失败: {e}")
        finally:
            self._global_config_loaded = True

    async def send_image_with_callback_api(self, image_path: str) -> Image:
        callback_api_base = self.context.get_config().get("callback_api_base")
        if not callback_api_base:
            return Image.fromFileSystem(image_path)
        try:
            image_component = Image.fromFileSystem(image_path)
            download_url = await image_component.convert_to_web_link()
            return Image.fromURL(download_url)
        except Exception as e:
            logger.warning(f"回退本地文件发送，原因: {e}")
            return Image.fromFileSystem(image_path)

    async def gemini_image_tool(self, event: AstrMessageEvent, image_description: str, use_reference_images: bool = True, mode: str = "auto"):
        """
        Generate or edit images via gcli2api endpoints.
        If images exist in the message/reply and use_reference_images=True, will include them.
        mode: "auto" | "generate" | "edit". When "auto", edit if references provided else generate.
        """
        await self._load_global_config()

        # gcli2api 模式：仅需 gcli2api_api_password（默认 pwd），无需官方 API Key

        # 收集参考图片（当前消息与引用消息）
        input_images = []
        if use_reference_images and hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    try:
                        base64_data = await comp.convert_to_base64()
                        input_images.append(base64_data)
                    except Exception as e:
                        logger.warning(f"参考图片转 base64 失败: {e}")
                elif isinstance(comp, Reply) and comp.chain:
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            try:
                                base64_data = await reply_comp.convert_to_base64()
                                input_images.append(base64_data)
                            except Exception as e:
                                logger.warning(f"引用图片转 base64 失败: {e}")

        # 模式与端点选择（流式优先），编辑与生成均走 generateContent，仅差别为是否带参考图
        endpoint_path = self._STREAM_GEN_PATH if self.use_stream else self._GEN_PATH

        try:
            if self.use_stream:
                from .utils.gemini_images_api import generate_or_edit_image_gemini_stream
                image_url, image_path = await generate_or_edit_image_gemini_stream(
                    prompt=image_description,
                    api_keys=[self.gcli2api_api_password] if self.gcli2api_api_password else [""],
                    model=self.model_name,
                    api_base=self.api_base,
                    endpoint_path=endpoint_path,
                    input_images_b64=input_images,
                    max_retry_attempts=self.max_retry_attempts,
                )
                # 流式失败则回退非流式
                if not image_path:
                    from .utils.gemini_images_api import generate_or_edit_image_gemini
                    image_url, image_path = await generate_or_edit_image_gemini(
                        prompt=image_description,
                        api_keys=[self.gcli2api_api_password] if self.gcli2api_api_password else [""],
                        model=self.model_name,
                        api_base=self.api_base,
                        endpoint_path=self._GEN_PATH,
                        input_images_b64=input_images,
                        max_retry_attempts=self.max_retry_attempts,
                    )
            else:
                from .utils.gemini_images_api import generate_or_edit_image_gemini
                image_url, image_path = await generate_or_edit_image_gemini(
                    prompt=image_description,
                    api_keys=[self.gcli2api_api_password] if self.gcli2api_api_password else [""],
                    model=self.model_name,
                    api_base=self.api_base,
                    endpoint_path=endpoint_path,
                    input_images_b64=input_images,
                    max_retry_attempts=self.max_retry_attempts,
                )

            if not image_path:
                yield event.plain_result("图像生成失败，请检查 API 配置与模型名称。")
                return

            # 可选：通过 Napcat 文件服务器中转
            if self.nap_server_address and self.nap_server_port:
                try:
                    new_path = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                    if new_path:
                        image_path = new_path
                except Exception as e:
                    logger.warning(f"Napcat 文件中转失败，回退为本地发送: {e}")

            image_component = await self.send_image_with_callback_api(image_path)
            yield event.chain_result([image_component])
        except Exception as e:
            logger.error(f"Gemini 生图/改图异常: {e}")
            yield event.plain_result(f"图像处理失败: {str(e)}")

    @filter.command_group("gimg")
    def gimg(self):
        """Gemini 生图/改图命令组"""
        pass

    @gimg.command("gen")
    async def cmd_generate(self, event: AstrMessageEvent, *, prompt: str):
        """生成图片：/gimg gen <描述>"""
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=False, mode="generate"):
            yield res

    @gimg.command("edit")
    async def cmd_edit(self, event: AstrMessageEvent, *, prompt: str):
        """根据消息中的图片改图：/gimg edit <修改描述>（需携带/引用图片）"""
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=True, mode="edit"):
            yield res

    # 已移除 gconf 指令组，配置请在 AstrBot 插件设置中修改
