library(ggdag)
library(dagitty)
library(GGally)


#identification for rainfall
dag_rain <- dagify(
  sir ~  Rain + SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + SoilMoi + ENSO + Temperature + forest + EVI + vectors + MPI + Year + Month + Municipality + Municipality_Year + Municipality_Year_Month,
  Rain ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + forest + ENSO +  Year + Month,
  SST12 ~ SST3 + SST34 + SST4 + NATL + SATL +  TROP + Year + Month,
  SST3 ~ SST34 + SST4 + NATL + SATL +  TROP + Year + Month,
  SST34 ~ SST4 + NATL + SATL +  TROP + Year + Month,
  SST4 ~ NATL + SATL +  TROP + Year + Month,
  NATL ~ SATL +  TROP + Year + Month,
  SATL ~  TROP +  Year + Month,
  TROP ~  Year + Month,
  ENSO ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + Year + Month,
  EVI ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP  + forest + ENSO + Temperature + Rain + SoilMoi + Year + Month,
  Temperature ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + forest + ENSO + Rain + SoilMoi + Year + Month,
  SoilMoi ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + forest + ENSO + Rain + Year + Month,
  vectors ~   SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + forest + ENSO + EVI + Rain + SoilMoi + Temperature + Year + Month,
  MPI ~ forest + EVI + Rain + SoilMoi + Temperature + vectors,
  
  exposure = "Rain",
  outcome = "sir",
  coords = list(x = c(sir=2.9, Rain =-1, vectors=1.1, SoilMoi=0.1, ENSO=-2.1, Temperature=0.8,
                      forest=-3.3, SST3=-1.8, SST4=-1.9, SST34=-2, SST12=-2.1, NATL=-2.2, SATL=-2.3,
                      TROP=-2.4, Year=-3.4, Month=-3.3, EVI=0.3, MPI=1.3, Municipality=1.9, Municipality_Year=1.8, Municipality_Year_Month=1.7),
                y = c(sir=0.5, Rain =0.5, vectors=1.5, SoilMoi=-1.6, ENSO=-2.5, Temperature=-1.2,
                      forest=4.5, SST3=1.3, SST4=1.4, SST34=1.5, SST12=1.6, NATL=1.7, SATL=1.8,
                      TROP=1.9, Year=-1.9, Month=-1.8, EVI=3.8, MPI=3.9, Municipality=-1.9, Municipality_Year=-1.8, Municipality_Year_Month=-1.7)))

# theme_dag
ggdag(dag_rain) + 
  theme_dag()

ggdag_status(dag_rain) +
  theme_dag()

#adjust
adjustmentSets(dag_rain,  type = "canonical")

ggdag_adjustment_set(dag_rain, type = "canonical", shadow = TRUE) +
  theme_dag()



#######################################
#identification for temperature
dag_temp <- dagify(
  sir ~  Temperature + SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + SoilMoi + ENSO + Rain + forest + EVI + vectors + MPI + Year + Month + Municipality + Municipality_Year + Municipality_Year_Month,
  Temperature ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + forest + ENSO + Rain + SoilMoi + Year + Month,
  SST12 ~ SST3 + SST34 + SST4 + NATL + SATL +  TROP + Year + Month,
  SST3 ~ SST34 + SST4 + NATL + SATL +  TROP + Year + Month,
  SST34 ~ SST4 + NATL + SATL +  TROP + Year + Month,
  SST4 ~ NATL + SATL +  TROP + Year + Month,
  NATL ~ SATL +  TROP + Year + Month,
  SATL ~  TROP +  Year + Month,
  TROP ~  Year + Month,
  ENSO ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + Year + Month,
  EVI ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP  + forest + ENSO + Temperature + Rain + SoilMoi + Year + Month,
  Rain ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + forest + ENSO + Year + Month,
  SoilMoi ~ SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + forest + ENSO + Rain + Year + Month,
  vectors ~   SST12 + SST3 + SST34 + SST4 + NATL + SATL + TROP + forest + ENSO + EVI + Rain + SoilMoi + Temperature + Year + Month,
  MPI ~ forest + EVI + Rain + SoilMoi + Temperature + vectors,
  
  exposure = "Temperature",
  outcome = "sir",
  coords = list(x = c(sir=2.9, Temperature =0.2, vectors=1.1, SoilMoi=-1.0, ENSO=-2.1, Rain=-1.3,
                      forest=-3.3, SST3=-1.8, SST4=-1.9, SST34=-2, SST12=-2.1, NATL=-2.2, SATL=-2.3,
                      TROP=-2.4, Year=-3.4, Month=-3.3, EVI=0.3, MPI=1.3, Municipality=1.9, Municipality_Year=1.8, Municipality_Year_Month=1.7),
                y = c(sir=0.5, Temperature =0.5, vectors=1.5, SoilMoi=-1.6, ENSO=-2.5, Rain=-2.2,
                      forest=4.5, SST3=1.3, SST4=1.4, SST34=1.5, SST12=1.6, NATL=1.7, SATL=1.8,
                      TROP=1.9, Year=-1.9, Month=-1.8, EVI=3.8, MPI=3.9, Municipality=-1.9, Municipality_Year=-1.8, Municipality_Year_Month=-1.7)))

# theme_dag
ggdag(dag_temp) + 
  theme_dag()

ggdag_status(dag_temp) +
  theme_dag()

#adjust
adjustmentSets(dag_temp,  type = "canonical")

ggdag_adjustment_set(dag_temp, type = "canonical", shadow = TRUE) +
  theme_dag()
