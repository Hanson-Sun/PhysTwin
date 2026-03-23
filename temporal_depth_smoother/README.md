$$\mathcal{L}_{\text{tgm}} = \mathcal{L}_{\text{tgm}}^{k=1} + 0.2 \cdot \mathcal{L}_{\text{tgm}}^{k=2} + 0.05 \cdot \mathcal{L}_{\text{tgm}}^{k=3}$$

$$\mathcal{L}_{\text{tv}} = \mathcal{L}_{\text{tv}}^{k=1} + 0.2 \cdot \mathcal{L}_{\text{tv}}^{k=2}$$
$$\mathcal{L} = \lambda_{\text{fid}} \cdot \mathcal{L}_{\text{fid}} + \lambda_{\text{tgm}} \cdot \mathcal{L}_{\text{tgm}} + \lambda_{\text{geom}} \cdot \mathcal{L}_{\text{geom}} + \lambda_{\text{tv}} \cdot \mathcal{L}_{\text{tv}}$$


$$\mathcal{L}_{\text{tgm}}^{k} = \frac{1}{B_{\text{tgm}}^{k}} \cdot \mathbb{E}\left[\left| \Delta_k \hat{d} - \Delta_k d_{\text{vda}} \right|\right]$$

$$\mathcal{L}_{\text{tv}}^{k} = \frac{1}{B_{\text{tv}}^{k}} \cdot \mathbb{E}\left[\left| \delta^2_k \hat{d} \right| \cdot \exp\left(-10 \cdot \left| \delta^2_k d_{\text{vda}} \right|\right)\right]$$

$$\mathcal{L}_{\text{fid}} = \frac{1}{B_{\text{fid}}} \cdot \mathbb{E}\left[\left| \hat{d}_{\text{aln}} - d_{\text{aln}} \right| \cdot g\right]$$

$$\mathcal{L}_{\text{geom}} = \frac{1}{2} \cdot \mathbb{E}\left[\left(\left|\nabla_x \hat{d} - \nabla_x d_{\text{raw}}\right| + \left|\nabla_y \hat{d} - \nabla_y d_{\text{raw}}\right|\right) \cdot w_{\text{edge}}\right]$$