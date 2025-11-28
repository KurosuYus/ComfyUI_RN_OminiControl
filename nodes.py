import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

class Kiki_Omini_Subject:
    @classmethod
    def INPUT_TYPES(s):
        import os
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
        except Exception:
            models_dir = os.environ.get('COMFYUI_MODELS_DIR', os.path.join(os.getcwd(), 'models'))
        flux_root = os.path.join(models_dir, 'flux')
        options = []
        if os.path.isdir(flux_root):
            for name in os.listdir(flux_root):
                full = os.path.join(flux_root, name)
                if os.path.isdir(full) and os.path.isdir(os.path.join(full, 'text_encoder')):
                    options.append(name)
        options = options or ["FLUX.1-schnell"]
        return {
            "required": {
                "subject_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ''}),
                "flux_model": (options, ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Subject'

    CATEGORY = "RunNode/Omini"
    DESCRIPTION = "Ominicontrol subject node"

    def run(self, subject_image, prompt, flux_model, seed):
        try:
            import ComfyUI_RN_OminiControl.rn_omini_subject as ros
        except Exception:
            pkg = __package__ or os.path.basename(os.path.dirname(os.path.realpath(__file__)))
            ros = importlib.import_module(f"{pkg}.rn_omini_subject")
        img = ros.run(subject_image, prompt, flux_model, seed)
        return (img, )
    
class Kiki_Omini_Spatial:
    @classmethod
    def INPUT_TYPES(s):
        import os
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
        except Exception:
            models_dir = os.environ.get('COMFYUI_MODELS_DIR', os.path.join(os.getcwd(), 'models'))
        flux_root = os.path.join(models_dir, 'flux')
        options = []
        if os.path.isdir(flux_root):
            for name in os.listdir(flux_root):
                full = os.path.join(flux_root, name)
                if os.path.isdir(full) and os.path.isdir(os.path.join(full, 'text_encoder')):
                    options.append(name)
        options = options or ["FLUX.1-schnell"]
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ''}),
                "condition_type": (["canny", "depth", "coloring", "deblurring"], ),
                "flux_model": (options, ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Spatial'

    CATEGORY = "RunNode/Omini"
    DESCRIPTION = "Ominicontrol spatial node"

    def run(self, ref_image, prompt, condition_type, flux_model, seed):
        try:
            import ComfyUI_RN_OminiControl.rn_omini_spatial as rosp
        except Exception:
            pkg = __package__ or os.path.basename(os.path.dirname(os.path.realpath(__file__)))
            rosp = importlib.import_module(f"{pkg}.rn_omini_spatial")
        img = rosp.run(ref_image, prompt, condition_type, flux_model, seed)
        return (img, )
    
class Kiki_Omini_Fill:
    @classmethod
    def INPUT_TYPES(s):
        import os
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
        except Exception:
            models_dir = os.environ.get('COMFYUI_MODELS_DIR', os.path.join(os.getcwd(), 'models'))
        flux_root = os.path.join(models_dir, 'flux')
        options = []
        if os.path.isdir(flux_root):
            for name in os.listdir(flux_root):
                full = os.path.join(flux_root, name)
                if os.path.isdir(full) and os.path.isdir(os.path.join(full, 'text_encoder')):
                    options.append(name)
        options = options or ["FLUX.1-schnell"]
        return {
            "required": {
                "ori_image": ("IMAGE",),
                "mask": ("MASK", ),
                "prompt": ("STRING", {"multiline": True, "default": ''}),
                "flux_model": (options, ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Fill'

    CATEGORY = "RunNode/Omini"
    DESCRIPTION = "Ominicontrol fill node"

    def run(self, ori_image, mask, prompt, flux_model, seed):
        try:
            import ComfyUI_RN_OminiControl.rn_omini_fill as rof
        except Exception:
            pkg = __package__ or os.path.basename(os.path.dirname(os.path.realpath(__file__)))
            rof = importlib.import_module(f"{pkg}.rn_omini_fill")
        img = rof.run(ori_image, mask, prompt, flux_model, seed)
        return (img, )

NODE_CLASS_MAPPINGS = {
    "RunNode_Omini_Subject": Kiki_Omini_Subject,
    "RunNode_Omini_Spatial": Kiki_Omini_Spatial,
    "RunNode_Omini_Fill": Kiki_Omini_Fill,
}
