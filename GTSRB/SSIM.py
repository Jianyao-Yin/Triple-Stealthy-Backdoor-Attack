from skimage.metrics import structural_similarity as ssim
import cv2

def SSIM(image1, image2, print_bool=1):
    image1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

    # 检查两张图像的尺寸是否一致
    if image1.shape != image2.shape:
        raise ValueError("两张图像的尺寸不一致，无法计算 SSIM")

    ssim_value, ssim_map = ssim(image1, image2, full=True)
    print(f"SSIM: {ssim_value}")
    if print_bool:
        plt.imshow(ssim_map, cmap='gray')
        plt.colorbar()
        plt.title("SSIM Map")
        plt.show()
