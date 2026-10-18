"""Default machines, surface equipment and base facilities. All numbers are planning assumptions, not vendor specs."""
from .models import TBM, Facility, HazardLimits, Vehicle

# First-base facilities as described for SpaceX's Starship plans. Where no source gives a size, the
# area is an assumption and says so. Sources:
#   Heldmann et al. 2022, "Mission Architecture Using the SpaceX Starship Vehicle to Enable a Sustained
#     Human Presence on Mars" (New Space; PMC9527650): uncrewed ships land ~1 km apart; prepared pads with
#     berms later; crews live in Starship for the first years; propellant plant (1,200 t per refill,
#     ~600 t water); solar first, nuclear later; water from ice.
#   Metzger 2021, "ISRU Implications for Lunar and Martian Plume Effects" (arXiv 2104.06248): ~1 km minimum
#     blast radius around an unprepared landing site.
#   Regolith-polymer landing pad study (NASA NTRS 20220000334): pads of 100 m or less fit one Starship payload.
#   ~2.55 MW for a two-ship propellant plant (Space Review / encyclopedia.pub summaries); ~40,000 m2 of
#     array per MW averaged over the Mars day (Inverse, after Zubrin's 1 MW estimate).
FACILITIES: dict[str, Facility] = {f.id: f for f in [
    Facility(id="landing-pad", name="Starship landing pad", area_m2=100 * 100, max_slope_deg=2, min_separation_m=1000,
             basis="100 m prepared pad (NTRS 20220000334); ships ~1 km apart and 1 km blast radius (Heldmann 2022; Metzger 2021)"),
    Facility(id="habitat", name="Habitat area", area_m2=60 * 60, max_slope_deg=3,
             basis="assumed 60 m x 60 m for modules, airlocks and shielded outdoor work area; crews start out living in landed Starships"),
    Facility(id="propellant-plant", name="Propellant plant (Sabatier + electrolysis)", area_m2=100 * 100, max_slope_deg=3,
             basis="assumed 100 m x 100 m for ~190 t of plant, cryogenic CH4/LOX tanks and water feed; makes 1,200 t per refill"),
    Facility(id="solar-farm", name="Solar farm, 1 MW", area_m2=200 * 200, max_slope_deg=8,
             basis="~40,000 m2 per MW averaged over the Mars day (Zubrin's 1 MW estimate); a two-ship plant needs ~2.5 MW"),
    Facility(id="ice-mine", name="Water ice extraction", area_m2=50 * 50, max_slope_deg=10,
             basis="assumed 50 m x 50 m Rodwell drill site; ~600 t of water per Starship refill"),
    Facility(id="depot", name="Cargo and storage yard", area_m2=50 * 50, max_slope_deg=5,
             basis="assumed 50 m x 50 m for unloaded cargo, spares and rovers"),
]}

# boring rates are set against a construction unit, which grades 600 m of road a sol on flat ground
# (construct/model.py): the main TBM bores a third of that, 200 m a sol after downtime (333 m x 0.6 duty)
TBMS: dict[str, TBM] = {t.id: t for t in [
    TBM(id="tbm-4m", name="Main-drive TBM, 4 m", diameter_m=4.0, advance_m_per_sol=333.0, duty=0.6,
        max_grade_pct=6.0, min_cover_m=5.0, swell=1.4, power_kw=300),
    TBM(id="tbm-2m", name="Compact TBM, 2 m", diameter_m=2.0, advance_m_per_sol=585.0, duty=0.65,
        max_grade_pct=10.0, min_cover_m=3.0, swell=1.4, power_kw=90),
]}

VEHICLES: dict[str, Vehicle] = {v.id: v for v in [
    Vehicle(id="crew-rover", name="Pressurized crew rover", width_m=3.5, payload_t=2, speed_kmh=10,
            limits=HazardLimits(max_grade_deg=18, max_cross_slope_deg=12, max_rock_m=0.4,
                                max_rock_cover_pct=3.0, min_crater_rim_clearance_m=200)),
    Vehicle(id="utility-rover", name="Light utility rover", width_m=2.2, payload_t=0.5, speed_kmh=6,
            limits=HazardLimits(max_grade_deg=20, max_cross_slope_deg=15, max_rock_m=0.35,
                                max_rock_cover_pct=5.0, min_crater_rim_clearance_m=0)),
    Vehicle(id="hauler", name="Regolith / ore hauler", width_m=4.0, payload_t=10, speed_kmh=4,
            limits=HazardLimits(max_grade_deg=12, max_cross_slope_deg=8, max_rock_m=0.25,
                                max_rock_cover_pct=4.0, min_crater_rim_clearance_m=300)),
    Vehicle(id="crane", name="Mobile crane", width_m=5.0, payload_t=20, hauls=False, speed_kmh=2,
            limits=HazardLimits(max_grade_deg=8, max_cross_slope_deg=5, max_rock_m=0.15,
                                max_rock_cover_pct=5.0, min_crater_rim_clearance_m=500)),
    Vehicle(id="tbm-carrier", name="TBM transporter", width_m=6.0, payload_t=60, hauls=False, speed_kmh=1,
            limits=HazardLimits(max_grade_deg=6, max_cross_slope_deg=4, max_rock_m=0.1,
                                max_rock_cover_pct=6.0, min_crater_rim_clearance_m=1000)),
]}
